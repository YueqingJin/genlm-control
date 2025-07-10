import re, pathlib, importlib.util

# Find the python.lark that comes with lark
src = pathlib.Path(importlib.util.find_spec("lark").origin).parent \
         / "grammars" / "python.lark"

# Write the simplified file to the same directory as the script
dst = pathlib.Path(__file__).with_name("python_simple.lark")

txt = src.read_text()

# Remove "lookahead"/" lookbehind"
edits = {
    # STRING (allows f r u prefix + single/triple quotes, requires closure) :
    r"STRING: /[fFrRuU]?(['\"]{3}|['\"]).*?(\1)/s"
        : r"STRING: /[fFrRuU]?(['\"]{3}|['\"]).*?(\1)/s",

    # BYTES (b prefix) :
    r"BYTES\s*:.*"
        : r"BYTES:  /[bB](['\"]{3}|['\"]).*?(\1)/s",

    # FSTRING STRING - the loosest:
    r"FSTRING_STRING:.*"
        : r"FSTRING_STRING: /.*?/",
}
for pat, rep in edits.items():
    txt = re.sub(pat, rep, txt)

if not txt.startswith("start:"):
    txt = "start: file_input\n" + txt

dst.write_text(txt)
print(f"Generated {dst.relative_to(pathlib.Path.cwd())}")
