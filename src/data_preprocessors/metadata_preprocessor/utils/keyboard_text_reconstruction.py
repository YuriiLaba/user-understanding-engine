class KeyboardTextReconstructor:
    """Reconstructs text from keyboard events."""

    def __init__(self):
        # Mapping for shifted symbols when Shift is pressed
        self.shift_symbols = {
            '`': '~', '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&', '8': '*', '9': '(', '0': ')', '-': '_',
            '=': '+', '[': '{', ']': '}', '\\': '|', ';': ':', "'": '"',
            ',': '<', '.': '>', '/': '?'
        }

        # Keys that should never directly appear in text
        self.ignored_keys = {
            'shift', 'right_shift', 'ctrl', 'right_ctrl', 'alt', 'right_alt',
            'cmd', 'right_cmd', 'caps_lock', 'fn', 'menu', 'meta', 'none'
        }

    def reconstruct(self, events: list[dict]) -> str:
        """Reconstruct text from keyboard events."""
        output: list[str] = []

        for e in events:
            # Only handle key press down events
            if e.get('event_type') != 'key_press':
                continue

            data = e.get('data', {}) or {}
            if data.get('action') != 'down':
                continue

            raw_char = str(data.get('key_char', '')).strip()
            modifiers = data.get('modifiers', {}) or {}

            # Modifier states
            is_cmd = (
                modifiers.get('cmd', False)
                or modifiers.get('right_cmd', False)
                or modifiers.get('meta', False)
            )
            is_ctrl = (
                modifiers.get('ctrl', False)
                or modifiers.get('right_ctrl', False)
            )
            is_alt = (
                modifiers.get('alt', False)
                or modifiers.get('right_alt', False)
            )
            is_shift = (
                modifiers.get('shift', False)
                or modifiers.get('right_shift', False)
            )
            is_caps = modifiers.get('caps_lock', False)

            # --- Deletion handling (incl. ctrl+backspace deletes word) ---
            if raw_char in ['backspace', 'delete']:
                if is_ctrl:
                    # Delete trailing spaces
                    while output and output[-1].isspace():
                        output.pop()
                    # Delete last word
                    while output and not output[-1].isspace():
                        output.pop()
                else:
                    if output:
                        output.pop()
                continue

            # Skip raw modifier keys themselves
            if raw_char in self.ignored_keys:
                continue

            # --- Shortcuts: cmd/ctrl/alt + key ---
            if is_cmd or is_ctrl or is_alt:
                # Common case: cmd/ctrl/alt + space (layout switch, spotlight, etc.)
                if raw_char == 'space':
                    # Ignore these; they are usually not "real text"
                    continue

                # Optionally record shortcuts in the output
                mods_used = []
                if is_cmd:
                    mods_used.append("cmd")
                if is_ctrl:
                    mods_used.append("ctrl")
                if is_alt:
                    mods_used.append("alt")
                if is_shift:
                    mods_used.append("shift")

                # Example representation: <ctrl+c>, <cmd+shift+s>
                shortcut_repr = "<" + "+".join(mods_used + [raw_char]) + ">"
                output.append(shortcut_repr)
                continue

            # --- Whitespace keys ---
            if raw_char == 'space':
                output.append(' ')
                continue
            elif raw_char in ['enter', 'return']:
                output.append('\n')
                continue
            elif raw_char == 'tab':
                output.append('\t')
                continue

            # --- Other special keys (arrows, escape, etc.) ---
            if len(raw_char) > 1:
                # Represent them explicitly; adjust if you prefer to skip
                output.append(f'<{raw_char}>')
                continue

            # --- Regular characters ---
            final_char = raw_char

            # Letters: apply shift + caps logic
            if final_char and len(final_char) == 1 and final_char.isalpha():
                # XOR: exactly one of shift/caps makes it uppercase
                final_char = final_char.upper() if (is_shift ^ is_caps) else final_char.lower()
            # Non-letter symbols where Shift changes the glyph (e.g., 1 → !)
            elif is_shift and final_char in self.shift_symbols:
                final_char = self.shift_symbols[final_char]

            if final_char:
                output.append(final_char)

        return "".join(output)
