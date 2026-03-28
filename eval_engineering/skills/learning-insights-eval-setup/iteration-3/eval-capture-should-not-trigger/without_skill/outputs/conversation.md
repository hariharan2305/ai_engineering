# Conversation: Rename .txt files to .md

**User:** Can you help me write a bash script that renames all .txt files in a directory to .md?

**Claude:** Sure. One-liner: `for f in *.txt; do mv "$f" "${f%.txt}.md"; done`

**User:** Perfect, that worked.

**User:** Can you save this conversation as a .md file in my project directory?
