# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded actor metadata protocol. No KV payload, generic exec, or SDK capability."""
import json
import re


def _unique(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ValueError('duplicate scenario metadata field')
        value[key]=item
    return value


def command(raw):
    if not raw or len(raw)>65536 or not raw.endswith('\n'):
        raise ValueError('bounded complete scenario frame required')
    row=json.loads(raw,object_pairs_hook=_unique)
    if (type(row) is not dict or set(row)!={'id','op','values'} or type(row['id']) is not int or row['id']<=0
            or type(row['values']) is not dict):
        raise ValueError('scenario envelope identity/schema differs')
    op=row['op'];v=row['values']
    schemas={'start':(set(),set()),'stop':(set(),set()),'prompt':(set(),{'label','size'}),
        'reset':(set(),{'avoid','probe_sentinels'}),'request':({'text'},{'interface'}),
        'capture':({'text','label'},{'guards'}),'overwrite':({'original'},set()),
        'route_observation':({'request_id','begin_ns','end_ns'},set())}
    if type(op) is not str or op not in schemas:raise ValueError('unsupported actor operation')
    required,optional=schemas[op]
    if not required<=set(v)<=required|optional:raise ValueError('scenario command fields differ')
    for key in ('text','label','request_id'):
        if key in v and (type(v[key]) is not str or not v[key] or len(v[key])>16384):
            raise ValueError('scenario text metadata differs')
    if 'label' in v and op=='capture' and not re.fullmatch('[A-Za-z0-9_-]{1,64}',v['label']):
        raise ValueError('capture label is not a bounded filename component')
    if 'interface' in v and v['interface'] not in ('direct','http'):raise ValueError('unsupported request interface')
    for key in ('guards','probe_sentinels'):
        if key in v and type(v[key]) is not bool:raise ValueError('guard flag must be boolean')
    if 'size' in v and (type(v['size']) is not int or v['size']!=257):raise ValueError('fixed257-token scenario required')
    if 'avoid' in v and (type(v['avoid']) is not list or len(v['avoid'])>2
            or any(type(n) is not int or n<0 for n in v['avoid'])):
        raise ValueError('bounded foreign source page-number exclusions required')
    for key in ('begin_ns','end_ns'):
        if key in v and (type(v[key]) is not int or v[key]<=0):raise ValueError('strict clock integer required')
    if 'begin_ns' in v and v['begin_ns']>v['end_ns']:raise ValueError('request clock interval reversed')
    if op=='overwrite' and (type(v['original']) is not dict or len(v['original'].get('blocks',[]))!=2
            or len(v['original'].get('prompt_ids',[]))!=257):
        raise ValueError('two contiguous complete original captures required')
    return row
