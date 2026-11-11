# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Install-only pending callback; fixed cooperative close signal, no arbitrary execution channel."""
try:
    import importlib.util
    import json
    import os
    from pathlib import Path
    import signal
    import time
    import traceback

    _root = Path(_UFLOW_RECOVERY_ROOT)  # noqa: F821 - fixed, hash-checked debugger bootstrap input
    _cfg = json.loads((_root / 'config.json').read_text())
    _ctx = {'stage': 'install', 'attempt': 0, 'busy': False, 'active_operation': None}
    _spec = importlib.util.spec_from_file_location('uflow_task_recovery_actions', Path(__file__).with_name('actions.py'))
    _actions = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_actions)

    def _write(name, data):
        with (_root / name).open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _event(stage):
        _ctx['stage'] = stage
        with (_root / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'stage': stage, 'pid': os.getpid(), 'birth': _actions.birth(os.getpid()),
                'attempt': _ctx['attempt'], 'operation_id': _ctx['active_operation'],
                'time_ns': time.monotonic_ns()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _cleanup(signum, current_frame):
        del current_frame
        if _ctx['busy']:
            return
        _ctx['busy'] = True
        _ctx['attempt'] += 1
        succeeded = False
        try:
            assert signum == signal.SIGUSR1 and os.getpid() == _cfg['pid']
            assert _actions.birth(os.getpid()) == _cfg['birth']
            if 'worker_close_receipt' not in _ctx:
                _event('worker-close')
                _actions.close_worker(_ctx, _cfg)
            if not _ctx.get('exports_released'):
                _event('release-unwound-exports')
                _actions.release_exports(_ctx)
                _ctx['exports_released'] = True
            if not _ctx.get('unregister_done'):
                _ctx['active_operation'] = _cfg['unregister_operation']
                _event('begin-unregister')
                _ctx['unregister_receipt'] = _ctx['client'].begin_unregister(operation_id=_cfg['unregister_operation'])
                _ctx['unregister_done'] = True
            _ctx['active_operation'] = None
            if not _ctx.get('root_released'):
                _event('release-root-memoryview')
                _ctx['access'].buffer.release()
                _ctx['root_released'] = True
            _event('actual-view-close')
            _ctx['client'].view.close()  # Real BufferError preserves the client; never ACK after it.
            _event('verify-no-pool-VMA-or-FD')
            _actions.verify_unmapped(_ctx, _cfg)
            if 'tracker_receipt' not in _ctx:
                _event('normal-tracker-stop')
                _ctx['tracker_receipt'] = _actions.stop_tracker(_ctx, _cfg)
            if not _ctx.get('ack_done'):
                _ctx['active_operation'] = _cfg['unmap_operation']
                _event('original-session-unmap-ACK')
                _ctx['ack_receipt'] = _ctx['client'].ack_unmapped(
                    native_drained=True, operation_id=_cfg['unmap_operation'])
                _ctx['ack_done'] = True
            _ctx['active_operation'] = None
            _event('fresh-Unmapped-snapshot')
            snapshot = _actions.verify_snapshot(_ctx, _cfg)
            _write('cleanup-success.json', {'pid': os.getpid(), 'birth': _actions.birth(os.getpid()),
                'mode': _cfg['mode'], 'worker_close': _ctx['worker_close_receipt'],
                'unmapped_observation': _ctx['unmapped_observation'], 'tracker': _ctx['tracker_receipt'],
                'unregister_receipt': _ctx['unregister_receipt'], 'ack_receipt': _ctx['ack_receipt'],
                'unmap_operation': _cfg['unmap_operation'], 'snapshot': snapshot,
                'scope': 'same original live process cooperatively closed; old dead-client pool untouched'})
            succeeded = True
        except BaseException as error:
            query = None
            if _ctx['active_operation'] is not None:
                try:
                    query = _ctx['client'].query_operation(_ctx['active_operation'])
                except BaseException as query_error:
                    query = {'query_error': repr(query_error)}
            try:
                _write(f"cleanup-failed-{_ctx['attempt']}.json", {'pid': os.getpid(),
                    'birth': _actions.birth(os.getpid()), 'stage': _ctx['stage'], 'error': repr(error),
                    'traceback': traceback.format_exc(), 'operation_id': _ctx['active_operation'], 'query': query,
                    'same_operation_retained': True, 'view_closed': _ctx.get('client') is not None and _ctx['client'].view.closed,
                    'ACK_done': _ctx.get('ack_done', False), 'original_objects_retained': True})
            except BaseException:
                pass
        finally:
            _ctx['busy'] = False
        if succeeded:
            raise SystemExit(0)  # Only after actual close/FD-unmap/ACK/snapshot; never on a failure.

    assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
    _prepared = _actions.prepare(_ctx, _cfg)
    signal.signal(signal.SIGUSR1, _cleanup)
    _write('handler-installed.json', {**_prepared, 'installed_only': True,
        'unmap_operation': _cfg['unmap_operation'], 'time_ns': time.monotonic_ns()})
except BaseException as _install_error:
    try:
        import json as _json
        import os as _os
        import traceback as _traceback
        from pathlib import Path as _Path
        with (_Path(_UFLOW_RECOVERY_ROOT) / 'install-failed.json').open('x') as _stream:  # noqa: F821
            _json.dump({'pid': _os.getpid(), 'error': repr(_install_error),
                'traceback': _traceback.format_exc(), 'pending_callback_error_swallowed': True,
                'original_process_retained': True}, _stream, indent=2)
    except BaseException:
        pass
