class ModeController:
    def __init__(self, default_mode="vital"):
        if default_mode not in ("vital", "alarm"):
            default_mode = "vital"
        self._mode = default_mode

    def current_mode(self):
        return self._mode

    def set_mode(self, m):
        if m not in ("vital", "alarm"):
            raise ValueError("mode must be 'vital' or 'alarm'")
        self._mode = m
