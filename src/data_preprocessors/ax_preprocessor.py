import json
import os
from typing import List, Dict, Tuple, Optional

class AXProcessor:
    """
    The 'Alignment-Safe' Version.
    1. Never drops a record (preserves ID alignment).
    2. 'Vacuum' strategy: recursive text extraction for unknown structures.
    3. Preserves Interactive & Layout roles.
    4. CAPTURES STATE (Checked, Selected, Expanded).
    """

    IGNORED_KEYS = {
        'id', 'nodeId', 'backendDOMNodeId', 'backendNodeId', 'parentId', 'frameId', 'childIds',
        'idx', 'index', 'pid', 'windowId', 'relation', 'relations', 'source', 'target',
        'bbox', 'rect', 'position', 'size', 'absolute_position', 'visible_bbox', 'clip',
        'x', 'y', 'width', 'height', 'top', 'left', 'bottom', 'right', 
        'color', 'backgroundColor', 'fgColor', 'bgColor', 'style', 'textStyle', 'layout',
        'is_pdf', 'superseded', 'busy', 'focusable', 'focused', 'editable', 
        'live', 'atomic', 'relevant', 'roledescription', 'level', 'posinset', 'setsize',
        'invalid', 'multiline', 'readonly' # Often noise, can be removed if strictly needed
    }

    INTERACTIVE_ROLES = {
        'button', 'input', 'textfield', 'search', 'link', 'checkbox', 'switch', 
        'slider', 'menu', 'menuitem', 'combobox', 'listbox', 'tab', 'tree', 'table',
        'axbutton', 'axlink', 'aximage', 'axcheckbox', 'axtextfield', 
        'scrollbar', 'splitter', 'toolbar'
    }

    # Boolean keys that represent critical UI state
    STATE_KEYS = {'checked', 'selected', 'expanded', 'pressed', 'disabled'}

    @staticmethod
    def load_ax(ax_path) -> List[Dict]:
        with open(ax_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        return data

    @staticmethod
    def _is_garbage_text(s: str) -> bool:
        # 1. Allow single digits (e.g., "1" on a calculator or calendar)
        if len(s) < 2:
            return not s.isdigit() and s not in {'+', '-', '=', '%', '$'}
        
        s_lower = s.lower()
        # 2. Filter obvious code noise
        if s_lower in {'true', 'false', 'none', 'null', 'undefined', 'div', 'span', 'body', 'view'}: return True
        # 3. Filter hex codes or generated IDs
        if s.startswith('#') and len(s) < 8: return True 
        return False

    @staticmethod
    def harvest_strings(node: Dict) -> Tuple[Optional[str], List[str]]:
        role = None
        found_text = []
        stack = [node]
        
        while stack:
            curr = stack.pop()
            if not isinstance(curr, dict): continue

            for k, v in curr.items():
                if k in AXProcessor.IGNORED_KEYS: continue
                
                # A. Capture Role
                if k.lower() in {'role', 'chromerole', 'axrole', 'role_description'}:
                    if not role or (isinstance(v, str) and len(v) < 20):
                        role = v
                    continue

                # B. Skip recursion into children (handled by main tree walker)
                if k in {'children', 'childIds', 'nodes', 'AXChildren'}:
                    continue

                # C. Capture Strings (The Vacuum)
                if isinstance(v, str):
                    s = v.strip()
                    if s and not AXProcessor._is_garbage_text(s):
                        if k in {'url', 'href', 'docUrl'} and s.startswith('http'):
                            found_text.append(f"URL: {s}")
                        elif s not in found_text:
                            found_text.append(s)
                
                # D. Capture Numbers (sliders, values)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                     if k.lower() in {'value', 'axvalue', 'valuenow'}:
                        found_text.append(str(v))
                
                # E. Capture State (Booleans) - NEW CRITICAL ADDITION
                elif isinstance(v, bool):
                    if v is True and k.lower() in AXProcessor.STATE_KEYS:
                        # Append explicit state marker, e.g., "[CHECKED]"
                        tag = f"[{k.upper()}]"
                        if tag not in found_text:
                            found_text.append(tag)

                # F. Recurse (Attribute Objects)
                elif isinstance(v, dict):
                    stack.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict): stack.append(item)
                        elif isinstance(item, str) and not AXProcessor._is_garbage_text(item):
                            if item not in found_text: found_text.append(item)

        return role, found_text

    @staticmethod
    def clean_and_compress(node: Dict, is_root: bool = False) -> Optional[Dict]:
        # 1. Visibility Check
        if node.get('visible') is False: return None
        if node.get('hidden') is True: return None

        # 2. Harvest Content
        role, content_list = AXProcessor.harvest_strings(node)
        
        clean_node = {}
        if role: clean_node['r'] = role
        if content_list:
            clean_node['d'] = " | ".join(dict.fromkeys(content_list))

        # 3. Process Children
        raw_children = (node.get('children') or node.get('childIds') or 
                        node.get('nodes') or node.get('AXChildren') or [])
        
        processed_children = []
        if isinstance(raw_children, list):
            for child in raw_children:
                if isinstance(child, dict):
                    res = AXProcessor.clean_and_compress(child, is_root=False)
                    if res: processed_children.append(res)

        # 4. Dedupe Child Echoes (Prevent: [Button] "Ok" -> [Text] "Ok")
        parent_text = clean_node.get('d', "")
        final_children = []
        for child in processed_children:
            child_text = child.get('d', "")
            child_role = str(child.get('r', "")).lower()
            
            # If child text matches parent text exactly, and child is not a complex container
            if child_text and parent_text and child_text == parent_text:
                if child_role == str(role).lower(): continue # Identical Role+Text
                if child_role in {'statictext', 'generic', 'text', 'group', 'image', 'img'}: continue # Redundant label
            
            final_children.append(child)
        processed_children = final_children

        # 5. Hoisting / Flattening
        has_content = 'd' in clean_node
        is_interactive = False
        if role:
            r_low = str(role).lower()
            is_interactive = any(x in r_low for x in AXProcessor.INTERACTIVE_ROLES)

        # A. Dead Leaf (No text, no children, not interactive)
        if not has_content and not processed_children:
            if is_root or is_interactive: return clean_node
            return None

        # B. Useless Wrapper (No text, not interactive, only 1 child) -> Return the child
        if not is_root and not is_interactive and not has_content and len(processed_children) == 1:
            return processed_children[0]

        if processed_children:
            clean_node['ch'] = processed_children

        return clean_node

    @staticmethod
    def to_string(node: Dict, depth=0) -> str:
        indent = "  " * depth
        role = node.get('r', 'Group')
        desc = f' "{node["d"]}"' if 'd' in node else ""
        lines = [f"{indent}[{role}]{desc}"]
        for child in node.get('ch', []):
            lines.append(AXProcessor.to_string(child, depth + 1))
        return "\n".join(lines)

    @staticmethod
    def process_ax(data: List[Dict], output_format='json') -> List[Dict]:
        final_output = []

        for i, row in enumerate(data):
            ax = row.get('accessibility', {})
            raw_root = ax.get('data') if isinstance(ax, dict) else None
            
            if not raw_root:
                final_output.append({'id': i, 'type': 'EMPTY', 'accessibility': '[Empty]'})
                continue
            
            # Handle List Roots (e.g. multiple windows)
            if isinstance(raw_root, list):
                if not raw_root:
                    final_output.append({'id': i, 'type': 'EMPTY', 'accessibility': '[Empty]'})
                    continue
                raw_root = raw_root[0] if len(raw_root) == 1 else {"role": "VirtualRoot", "children": raw_root}

            if not isinstance(raw_root, dict):
                final_output.append({'id': i, 'type': 'ERROR', 'accessibility': '[Error]'})
                continue

            # Process
            if raw_root.get('is_pdf'):
                summary = raw_root.get('pdf_summary', {})
                txt = str(summary.get('text', summary) if isinstance(summary, dict) else summary)
                clean_root = {"r": "PDF", "d": txt[:1500]} # Truncate massive PDF dumps
                type_label = "PDF"
            else:
                clean_root = AXProcessor.clean_and_compress(raw_root, is_root=True)
                type_label = 'BROWSER' if row.get('accessibility', {}).get('is_browser') else 'APP'
            
            payload = AXProcessor.to_string(clean_root) if output_format == 'string' and clean_root else clean_root
            if not payload: payload = "[Empty Tree]"
            
            final_output.append({
                'id': i,
                'type': type_label,
                'accessibility': payload
            })

        return final_output

    @staticmethod
    def save_ax(data_clean: List[Dict], save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_clean, f, ensure_ascii=False, indent=2)

    @staticmethod
    def process_and_save(ax_in_path: str, ax_out_path: str):

        if os.path.exists(ax_out_path):
            print(f"[INFO] Folder '{ax_out_path}' already exists. Skipping.")
            return

        data = AXProcessor.load_ax(ax_in_path)
        print(f"[INFO] Loaded {len(data)} AX records")
        clean_data = AXProcessor.process_ax(data, output_format='string')
        AXProcessor.save_ax(clean_data, ax_out_path)
        print(f"[INFO] Saved AX to: {ax_out_path}")
