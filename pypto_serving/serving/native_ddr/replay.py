# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FIFO delivery receipts, retired only after stop AND consumer acknowledgement.

Sequences belong to actual delivery, not logical job creation. Replaying an old
envelope below the retired frontier cannot execute even after its payload was
forgotten. Acknowledgements cannot cross pending/unknown/protected records.
"""
from dataclasses import dataclass, replace

REPLAY_WINDOW = 16
MAX_UNACKNOWLEDGED = 64


class ReplayError(ValueError):
    def __init__(self, message, *, stopped=True):
        super().__init__(message)
        self.stopped = stopped


@dataclass
class Receipt:
    identity: object
    response: object = None
    protected: bool = False


class DeliverySender:
    def __init__(self):
        self.sent = 0
        self.acknowledged = 0
        self._completed = set()

    @property
    def available(self):
        return self.sent - self.acknowledged < MAX_UNACKNOWLEDGED

    def stamp(self, request):
        if not self.available:
            raise RuntimeError("delivery acknowledgement window is full")
        self.sent += 1
        return replace(request, delivery_sequence=self.sent, retire_through=self.acknowledged)

    def consume(self, sequence):
        if type(sequence) is not int or not 0 < sequence <= self.sent:
            raise ValueError("receipt is not for a delivered command")
        if sequence <= self.acknowledged:
            return
        self._completed.add(sequence)
        while self.acknowledged + 1 in self._completed:
            self.acknowledged += 1
            self._completed.remove(self.acknowledged)


class ReplayLedger:
    def __init__(self):
        self.high_water = 0
        self.acknowledged = 0
        self.retired_through = 0
        self.records = {}
        self._identities = {}

    def begin(self, request, identity_key):
        sequence, ack = request.delivery_sequence, request.retire_through
        if type(sequence) is not int or sequence <= 0 or type(ack) is not int or ack < 0:
            raise ReplayError("invalid delivery sequence/acknowledgement")
        if ack > self.high_water:
            raise ReplayError("acknowledgement exceeds delivered frontier", stopped=False)
        # A replay carries its original (possibly older) watermark. It cannot
        # roll the frontier back. First validate the WHOLE advance, then mutate.
        if ack > self.acknowledged:
            for index in range(self.acknowledged + 1, ack + 1):
                record = self.records[index]
                if record.response is None or record.protected:
                    raise ReplayError("acknowledgement crosses retained native/control obligation", stopped=False)
            self.acknowledged = ack
            frontier = max(self.retired_through, ack - REPLAY_WINDOW)
            for index in range(self.retired_through + 1, frontier + 1):
                old = self.records.pop(index)
                del self._identities[self._key(old.identity)]
            self.retired_through = frontier
        if sequence <= self.retired_through:
            raise ReplayError("retired delivery sequence; replay rejected before effects")
        record = self.records.get(sequence)
        if record is not None:
            if record.identity != request:
                raise ReplayError("delivery identity replay conflict", stopped=False)
            return record, False
        if sequence != self.high_water + 1:
            raise ReplayError("non-FIFO delivery sequence")
        if identity_key in self._identities:
            raise ReplayError("logical identity reused with a new delivery sequence", stopped=False)
        if self.high_water - self.acknowledged >= MAX_UNACKNOWLEDGED:
            raise ReplayError("unacknowledged delivery window full")
        record = Receipt(request)
        self.records[sequence] = record
        self._identities[identity_key] = sequence
        self.high_water = sequence
        return record, True

    @staticmethod
    def _key(identity):
        return getattr(identity, 'job_id', None) if not hasattr(identity, 'command_id') else identity.command_id

    def finish(self, request, response, *, protected=False):
        record = self.records.get(request.delivery_sequence)
        if record is not None and record.identity == request:
            record.response = response
            record.protected = protected

    def stats(self):
        return {'high_water': self.high_water, 'acknowledged': self.acknowledged,
                'retired_through': self.retired_through, 'records': len(self.records),
                'pending_or_protected': sum(r.response is None or r.protected for r in self.records.values()),
                'replay_window': REPLAY_WINDOW, 'max_unacknowledged': MAX_UNACKNOWLEDGED}
