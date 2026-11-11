# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Only install a fixed read-only n005 diagnostic handler; errors retain the original pause."""
try:
    import importlib.util
    import json
    import os
    from pathlib import Path
    import signal
    import traceback

    _root = Path(_UFLOW_RECOVERY_ROOT)  # noqa: F821
    _cfg = json.loads((_root / 'config.json').read_text())
    _ctx = {'busy': False, 'attempt': 0}
    _spec = importlib.util.spec_from_file_location('uflow_n005_diagnostic', Path(__file__).with_name('actions.py'))
    _actions = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_actions)

    def _write(name, value):
        with (_root / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def _diagnostic_signal(signum, current_frame):
        del current_frame
        if signum != signal.SIGUSR1 or _ctx['busy']:
            return
        _ctx['busy'], _ctx['attempt'] = True, _ctx['attempt'] + 1
        _exit_after_host_unmap = False
        try:
            if _cfg.get('action') == 'close-existing-device':
                assert _ctx['attempt'] == 1
                _write('before-device-close-diagnostic.json', _actions.diagnose(_ctx, _cfg))
                _actions.close_device(_ctx, _cfg, _write)
            elif _cfg.get('action') == 'finish-host-unmap':
                assert _ctx['attempt'] == 1
                _actions.finish_host_unmap(_ctx, _cfg, _write, _previous_namespace)  # noqa: F821
                _exit_after_host_unmap = True
            else:
                _write(f"diagnostic-{_ctx['attempt']}.json", _actions.diagnose(_ctx, _cfg))
        except BaseException as error:
            try:
                _operation = _ctx.get('terminal_state', {}).get('operation_id')
                _query = None
                if _operation is not None:
                    try:
                        _query = _ctx['client'].query_operation(_operation)
                    except BaseException as _query_error:
                        _query = {'error': repr(_query_error)}
                _write(f"diagnostic-failed-{_ctx['attempt']}.json", {'pid': os.getpid(), 'error': repr(error),
                    'traceback': traceback.format_exc(), 'original_context_retained': True,
                    'stage': _ctx.get('terminal_state', {}).get('stage', _ctx.get('stage', 'diagnosis')),
                    'operation_id': _operation, 'original_operation_query': _query,
                    'device_close_attempted': _cfg.get('action') == 'close-existing-device'})
            except BaseException:
                pass
        finally:
            _ctx['busy'] = False
        if _exit_after_host_unmap:
            raise SystemExit(0)  # Only after original SDK ACK + fresh Invalid/Unmapped proof.

    _prepared = _actions.prepare(_ctx, _cfg)
    _previous = signal.getsignal(signal.SIGUSR1)
    if _previous != _ctx['original']._request:
        assert _previous.__name__ == '_diagnostic_signal'
        assert _previous.__globals__['__file__'] == _cfg['previous_payload']
        assert _previous.__globals__['_root'] == Path(_cfg['previous_root'])
        assert _previous.__globals__['_cfg']['pid'] == _cfg['pid']
        assert _previous.__globals__['_ctx']['original'] is _ctx['original']
        import hashlib
        assert hashlib.sha256(Path(_cfg['previous_payload']).read_bytes()).hexdigest() == _cfg['previous_payload_sha256']
    signal.signal(signal.SIGUSR1, _diagnostic_signal)
    _write('handler-installed.json', _prepared)
except BaseException as _install_error:
    try:
        import json as _json
        import os as _os
        import traceback as _traceback
        from pathlib import Path as _Path
        (_Path(_UFLOW_RECOVERY_ROOT) / 'install-failed.json').write_text(_json.dumps({  # noqa: F821
            'pid': _os.getpid(), 'error': repr(_install_error), 'traceback': _traceback.format_exc(),
            'pending_callback_error_swallowed': True, 'original_process_retained': True}))
    except BaseException:
        pass
