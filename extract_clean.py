"""Quick extractor: pull the clean transcript out of a botched JSON wrap."""
import re
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Get the CLEAN VERSION section
parts = content.split('CLEAN VERSION')
if len(parts) > 1:
    inner = parts[1].split('-' * 70, 1)[1].strip()
else:
    inner = content

# The inner content is JSON-like but may have literal newlines inside strings,
# which makes json.loads fail. Use forgiving regex to pull the two fields out.

# Match "assessment": "..." up to the next field marker.
assess_match = re.search(
    r'"assessment"\s*:\s*"([\s\S]*?)"\s*,\s*"clean_version"',
    inner,
)
assessment = assess_match.group(1) if assess_match else ''

# Match "clean_version": "..." — either up to next field, or all remaining
# content if Claude's output got truncated before reaching the next field.
clean_match = re.search(
    r'"clean_version"\s*:\s*"([\s\S]*?)"\s*,\s*"alternatives"',
    inner,
)
if not clean_match:
    # Truncated output — grab everything from "clean_version" to end
    clean_match = re.search(
        r'"clean_version"\s*:\s*"([\s\S]*)$',
        inner,
    )
clean_transcript = clean_match.group(1) if clean_match else ''

# If we grabbed-to-end, strip any trailing JSON syntax noise
clean_transcript = clean_transcript.rstrip('",}\n ')

# Decode escape sequences like \n, \", \\
def unescape(s):
    return (s
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace('\\\\', '\\'))

clean_transcript = unescape(clean_transcript)
assessment = unescape(assessment)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('BANGLISH INTERPRETER -- CHAT-READY TRANSCRIPT\n')
    f.write('=' * 70 + '\n\n')
    f.write('ASSESSMENT:\n')
    f.write(assessment + '\n\n')
    f.write('=' * 70 + '\n')
    f.write('CLEANED TRANSCRIPT\n')
    f.write('=' * 70 + '\n\n')
    f.write(clean_transcript + '\n')

print(f'Saved: {output_path}')
print(f'Transcript length: {len(clean_transcript)} chars')
