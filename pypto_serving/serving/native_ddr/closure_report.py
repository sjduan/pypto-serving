# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Classify physical closure separately from SDK and DDR obligations."""

def classify(processes, resources, logical):
    """No successful leader exit or fake ACK substitutes for an empty proven subtree/driver."""
    tree_closed = (processes.get('process_tree_empty') is True and not processes.get('remaining')
                   and not processes.get('isolation_error') and type(processes.get('actual_task_exit_code')) is int)
    physical = tree_closed and resources.get('physical_closed') is True
    forced = any(e.get('event') == 'signal' and e.get('sent') and e.get('signal') in ('SIGTERM', 'SIGKILL')
                 for e in processes.get('events', []))
    if not physical:
        status = 'device-quarantined'
    elif (processes.get('evidence_errors') or processes.get('private_resources', {}).get('cleanup_complete') is False):
        status = 'cleanup-failed'
    elif logical.get('status') == 'sdk-closed':
        status = 'forced-physical-closed-sdk-confirmed' if forced else 'graceful-sdk-closed'
    elif logical.get('status') == 'not-used':
        status = 'forced-physical-closed-no-DDR' if forced else 'graceful-no-DDR'
    else:
        status = 'killed-physical-closed-ddr-unknown' if forced else 'physical-closed-ddr-unknown'
    return {'status': status, 'physical_device_reusable': physical, 'logical_DDR_status': logical.get('status', 'ddr-unknown'),
            'logical_DDR_budget_retained': logical.get('status') not in ('sdk-closed', 'not-used'),
            'shared_pool_budget_unchanged': True, 'pool_Destroy_attempted': False,
            'actual_task_exit_code': processes.get('actual_task_exit_code'), 'process_tree_empty': tree_closed,
            'SDK_ACK_inferred_from_death': False, 'physical': resources, 'logical': logical, 'processes': processes}
