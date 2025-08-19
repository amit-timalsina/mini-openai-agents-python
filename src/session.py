from openai.types.responses import ResponseInputItemParam


class Session:
    """Manages conversation history for an agent session.

    Stores messages in chronological order to maintain context
    without requiring explicit memory management.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[ResponseInputItemParam] = []

    def get_items(self, limit: int | None = None) -> list[ResponseInputItemParam]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history
        """
        if limit is None:
            return self.messages[:]
        return self.messages[-limit:]

    def add_items(self, items: list[ResponseInputItemParam]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        self.messages.extend(items)

    def pop_item(self) -> ResponseInputItemParam | None:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        if not self.messages:
            return None
        return self.messages.pop()

    def clear_session(self) -> None:
        """Clear all items for this session."""
        self.messages.clear()
