with open('ui/app.js', 'r', encoding='utf-8') as f:
    text = f.read()

in_str = None
in_s_cmt = False
in_m_cmt = False
stack = []
line_num = 1
i = 0
n = len(text)
while i < n:
    c = text[i]
    if c == '\n':
        line_num += 1
        in_s_cmt = False
        i += 1
        continue
    if in_s_cmt:
        i += 1
        continue
    if in_m_cmt:
        if c == '*' and i+1 < n and text[i+1] == '/':
            in_m_cmt = False
            i += 2
        else:
            i += 1
        continue
    if in_str:
        if c == '\\':
            i += 2
            continue
        if c == in_str:
            in_str = None
        i += 1
        continue
    if c == '/' and i+1 < n and text[i+1] == '/':
        in_s_cmt = True
        i += 2
        continue
    if c == '/' and i+1 < n and text[i+1] == '*':
        in_m_cmt = True
        i += 2
        continue
    if c in ('"', "'", '`'):
        in_str = c
        i += 1
        continue
    if line_num == 2247:
        print("Stack at line 2247 before processing line 2247:", stack)
        break
    if c == '{':
        stack.append((line_num, c))
    elif c == '}':
        if stack:
            stack.pop()
    i += 1
