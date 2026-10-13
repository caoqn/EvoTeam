from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)
    read: bool = False
    handoff_rule_id: str | None = None


class MessageStore:

    def __init__(self):
        self._messages: list[Message] = []
        self._counter: int = 0
        self._wake_events: dict[str, asyncio.Event] = {}
        self._terminate_event = asyncio.Event()
        self._terminate_reason: str | None = None

    def register(self, agent_name: str) -> None:
        if agent_name not in self._wake_events:
            self._wake_events[agent_name] = asyncio.Event()

    def unregister(self, agent_name: str) -> None:
        self._wake_events.pop(agent_name, None)

    def send(
        self, sender: str, receiver: str, content: str,
        handoff_rule_id: str | None = None,
    ) -> Message:
        if receiver not in self._wake_events and receiver != "[SYSTEM]":
            import logging
            logging.getLogger(__name__).warning(
                f"MessageStore.send: receiver '{receiver}' is not registered. "
                f"Registered agents: {list(self._wake_events.keys())}. "
                f"Message from '{sender}' will be stored but may never be read."
            )

        self._counter += 1
        msg = Message(
            id=f"msg_{self._counter}",
            sender=sender,
            receiver=receiver,
            content=content,
            handoff_rule_id=handoff_rule_id,
        )
        self._messages.append(msg)

        event = self._wake_events.get(receiver)
        if event is not None:
            event.set()

        return msg

    def read(
        self,
        reader: str,
        sender: str | None = None,
        unread_only: bool = True,
        limit: int = 20,
    ) -> list[Message]:
        results: list[Message] = []
        for msg in self._messages:
            if msg.receiver != reader:
                continue
            if sender and msg.sender != sender:
                continue
            if unread_only and msg.read:
                continue
            results.append(msg)
            if len(results) >= limit:
                break

        for msg in results:
            msg.read = True

        event = self._wake_events.get(reader)
        if event is not None:
            if self.count_unread(reader) == 0:
                event.clear()

        return results

    def count_unread(self, agent_name: str) -> int:
        return sum(
            1 for msg in self._messages
            if msg.receiver == agent_name and not msg.read
        )

    def count_unread_from(self, agent_name: str, from_agents: list[str]) -> int:
        from_set = set(from_agents)
        return sum(
            1 for msg in self._messages
            if msg.receiver == agent_name
            and msg.sender in from_set
            and not msg.read
        )

    def peek(
        self,
        reader: str,
        sender: str | None = None,
        unread_only: bool = False,
        limit: int = 200,
    ) -> list[Message]:
        results: list[Message] = []
        for msg in self._messages:
            if msg.receiver != reader:
                continue
            if sender and msg.sender != sender:
                continue
            if unread_only and msg.read:
                continue
            results.append(msg)
            if len(results) >= limit:
                break
        return results

    def count_from(self, sender: str, receiver: str) -> dict:
        total = 0
        unread = 0
        for msg in self._messages:
            if msg.sender == sender and msg.receiver == receiver:
                total += 1
                if not msg.read:
                    unread += 1
        return {"total": total, "unread": unread, "read": total - unread}

    async def wait_for_message(
        self, agent_name: str, timeout: float = 30.0,
        from_agents: list[str] | None = None,
    ) -> bool:
        event = self._wake_events.get(agent_name)
        if event is None:
            return False

        if self.count_unread(agent_name) > 0:
            return True

        import time as _time
        deadline = _time.monotonic() + timeout
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            if self._terminate_event.is_set():
                return True

            if self.count_unread(agent_name) > 0:
                return True

            event.clear()

    @property
    def registered_agents(self) -> list[str]:
        return list(self._wake_events.keys())


    def terminate(self, reason: str = ""):
        if self._terminate_reason is None:
            self._terminate_reason = reason
        self._terminate_event.set()
        for event in self._wake_events.values():
            event.set()

    def reset_termination(self) -> None:
        self._terminate_event.clear()
        self._terminate_reason = None
        for event in self._wake_events.values():
            event.clear()

    def is_terminated(self) -> bool:
        return self._terminate_event.is_set()

    @property
    def terminate_reason(self) -> str:
        return self._terminate_reason or ""

    def get_communication_partners(self, agent_name: str) -> list[str]:
        partners: set[str] = set()
        for msg in self._messages:
            if msg.sender == agent_name and msg.receiver != "[SYSTEM]":
                partners.add(msg.receiver)
            elif msg.receiver == agent_name and msg.sender != "[SYSTEM]":
                partners.add(msg.sender)
        return sorted(partners)

    def get_pairwise_initiators(self, agents: list[str]) -> dict[tuple[str, str], str]:
        agent_set = set(agents)
        first_sender: dict[tuple[str, str], str] = {}

        for msg in self._messages:
            s, r = msg.sender, msg.receiver
            if s not in agent_set or r not in agent_set:
                continue
            if s == r:
                continue
            pair = (min(s, r), max(s, r))
            if pair not in first_sender:
                first_sender[pair] = s

        return first_sender

    def iter_all_messages(self) -> list["Message"]:
        return list(self._messages)

    @property
    def total_count(self) -> int:
        return len(self._messages)