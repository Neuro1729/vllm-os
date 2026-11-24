# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def explicit_band_of(request: Request, default_band: int = 2) -> int:
    """Get the independently selected band from request sampling metadata."""
    params = request.sampling_params
    extras = getattr(params, "extra_args", None) if params is not None else None
    band = (extras or {}).get("priority_band", default_band)
    if type(band) is not int or band not in (1, 2):
        raise ValueError("priority_band must be exactly integer 1 or 2")
    return band


def explicit_band_key(
    request: Request,
    fairness_horizon_ms: int,
    reference_gap: int = 7,
    default_band: int = 2,
) -> tuple[int, int, float, str, int]:
    """Band-first, immutable waiting-order key; lower tuple wins.

    Band 2 equivalent score: arrival_time + priority * (H_ms/G/1000).
    We use integer microseconds to preserve ordering for large integer P.
    Shared -now term cancels under uniform, fixed-rate linear aging.
    """
    if fairness_horizon_ms <= 0 or reference_gap <= 0:
        raise ValueError("fairness horizon and reference gap must be positive")
    band = explicit_band_of(request, default_band)
    arrival = request.arrival_time
    if band == 1:
        return (0, request.priority, arrival, request.request_id, id(request))

    # Compare G * arrival_microseconds + P * H_microseconds. This is
    # monotonically equivalent to the uncapped linear-aging score.
    arrival_us = round(arrival * 1_000_000)
    score = reference_gap * arrival_us + request.priority * fairness_horizon_ms * 1000
    return (1, score, arrival, request.request_id, id(request))


def explicit_band_victim_key(
    request: Request, default_band: int = 2
) -> tuple[int, int, float]:
    """Prefer Band 2 as KV-pressure victim; original priority within band.

    Running-request priority does not acquire a waiting-time boost. This
    is a victim-selection policy, not a new preemption trigger.
    """
    return (
        explicit_band_of(request, default_band),
        request.priority,
        request.arrival_time,
    )


class ExplicitBandPriorityRequestQueue(RequestQueue):
    """Static-key heap for protected Band 1 and aging Band 2."""

    def __init__(
        self,
        fairness_horizon_ms: int,
        reference_gap: int = 7,
        default_band: int = 2,
    ) -> None:
        if fairness_horizon_ms <= 0 or reference_gap <= 0:
            raise ValueError("fairness horizon and reference gap must be positive")
        if default_band not in (1, 2):
            raise ValueError("default band must be 1 or 2")
        self.fairness_horizon_ms = fairness_horizon_ms
        self.reference_gap = reference_gap
        self.default_band = default_band
        self._heap: list[tuple[tuple[int, int, float, str, int], Request]] = []

    def key(self, request: Request) -> tuple[int, int, float, str, int]:
        return explicit_band_key(
            request, self.fairness_horizon_ms, self.reference_gap,
            self.default_band,
        )

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, (self.key(request), request))

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty band queue")
        return heapq.heappop(self._heap)[1]

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty band queue")
        return self._heap[0][1]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        for i, (_, queued) in enumerate(self._heap):
            if queued is request:
                del self._heap[i]
                heapq.heapify(self._heap)
                return
        raise ValueError("request not in band queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        removed = requests if isinstance(requests, set) else set(requests)
        self._heap = [
            (key, request) for key, request in self._heap if request not in removed
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        copy = self._heap[:]
        while copy:
            yield heapq.heappop(copy)[1]


def create_request_queue(
    policy: SchedulingPolicy,
    *,
    priority_band_fairness_horizon_ms: int = 0,
    priority_band_reference_gap: int = 7,
    priority_band_default: int = 2,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        if priority_band_fairness_horizon_ms > 0:
            return ExplicitBandPriorityRequestQueue(
                priority_band_fairness_horizon_ms,
                priority_band_reference_gap,
                priority_band_default,
            )
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
