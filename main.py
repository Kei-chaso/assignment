from pathlib import Path
import html
import hashlib
import json
import re
import time

from gemini.gemini_api_call import AttachmentConfig, GeminiRequestConfig, GeminiRequestExecutor, GenerationControls

def add_new_vocab(old_vocab_list_path: Path, additional_vocab_list_path: Path):
    old_vocab_text = old_vocab_list_path.read_text(encoding="utf-8")
    json_data = json.loads(old_vocab_text) if old_vocab_text else []
    normalized_old_vocab_set = {item["vocab"].strip().casefold() for item in json_data if "vocab" in item and isinstance(item["vocab"], str)}
    additional_vocab_list = set(additional_vocab_list_path.read_text(encoding="utf-8").splitlines())
    new_vocab_list = []
    for item in additional_vocab_list:
        item = item.strip()
        normalized_item = item.strip().casefold()
        if normalized_item and normalized_item not in normalized_old_vocab_set:
            normalized_old_vocab_set.add(normalized_item)
            new_vocab_list.append(item)
            json_data.append(
                {
                    "vocab": item,
                    "definition": None,
                    "provideContext": False,
                    "userPrompt": None,
                }
            )
    if new_vocab_list:
        print(f"New vocabs: {new_vocab_list}")
    return json_data

def _normalize_generated_definition(text: str) -> str:
    cleaned_text = text.strip()
    if "\n" not in cleaned_text and "\r" not in cleaned_text:
        return cleaned_text
    return " ".join(cleaned_text.split())

def reset_definitions(json_data: list[dict], reset_vocab_list_path: Path | None = None):
    reset_vocabs = []
    if not reset_vocab_list_path:
        for item in json_data:
            item["definition"] = None
            reset_vocabs.append(item["vocab"])
    else:
        reset_vocab_list = reset_vocab_list_path.read_text(encoding="utf-8").splitlines()
        normalized_reset_vocab_list = [item.strip().casefold() for item in reset_vocab_list if item.strip()]
        for item in json_data:
            vocab = item.get("vocab", "")
            if vocab.strip().casefold() in normalized_reset_vocab_list:
                item["definition"] = None
                reset_vocabs.append(vocab)
    reset_vocab_list_path.write_text("", encoding="utf-8")
    if reset_vocabs:
        print(f"Reset vocabs for definition: {reset_vocabs}")
    return json_data

def add_new_sentences(old_sentences_path: Path, additional_sentences_path: Path):
    additional_sentences = additional_sentences_path.read_text(encoding="utf-8").splitlines()
    old_sentences_text = old_sentences_path.read_text(encoding="utf-8")
    old_sentences_list = json.loads(old_sentences_text) if old_sentences_text else []
    normalized_old_sentences_keys = {
        item["sentence"].strip().casefold()
        for item in old_sentences_list
    }
    new_sentence_list = []
    for sentence in additional_sentences:
        sentence = sentence.strip()
        normalized_sentence = sentence.casefold()
        if normalized_sentence and normalized_sentence not in normalized_old_sentences_keys:
            normalized_old_sentences_keys.add(normalized_sentence)
            new_sentence_list.append(sentence)
            old_sentences_list.append(
                {
                    "sentence": sentence, 
                    "footnote": None,
                    "provideContext": False,
                    "userPrompt": None
                }
            )
    if new_sentence_list:
        print(f"New sentences: {new_sentence_list}")
    return old_sentences_list

def reset_footnote(json_data: list[dict], reset_sentence_path: Path | None = None):
    reset_sentences = []
    if reset_sentence_path is None:
        for item in json_data:
            item["footnote"] = None
            reset_sentences.append(item["sentence"])
    else:
        reset_sentence_list = reset_sentence_path.read_text(encoding="utf-8").splitlines()
        normalized_reset_sentence_list = [item.strip().casefold() for item in reset_sentence_list if item.strip()]
        for item in json_data:
            sentence = item.get("sentence", "")
            if sentence.strip().casefold() in normalized_reset_sentence_list:
                item["footnote"] = None
                reset_sentences.append(sentence)
    if reset_sentences:
        print(f"Reset sentences for footnote: {reset_sentences}")
    return json_data

def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def _print_generation_progress(label: str, processed: int, total: int, started_at: float):
    if total <= 0:
        print(f"{label}: 0/0 (100.0%) [########################] ETA 00:00")
        return
    percent = processed / total
    filled = int(24 * percent)
    bar = "#" * filled + "-" * (24 - filled)
    elapsed = time.monotonic() - started_at
    eta = (elapsed / processed) * (total - processed) if processed else None
    print(f"{label}: {processed}/{total} ({percent * 100:5.1f}%) [{bar}] ETA {_format_duration(eta)}")

def get_definitions(
        json_data: list[dict], 
        model_name: str = "gemini-3.1-flash-lite-preview", 
        system_instruction: str = "", 
        provide_context : bool = False,   
        context_file_path: Path | None = None):
    executor = GeminiRequestExecutor()
    attachments = []
    if context_file_path is not None:
        attachments.append(
            AttachmentConfig(
                file_path=context_file_path,
                display_name=context_file_path.name,
            )
        )
    pending_total = sum(1 for item in json_data if item.get("definition") is None)
    processed = 0
    started_at = time.monotonic()
    _print_generation_progress("Definition generation", processed, pending_total, started_at)
    for item in json_data:
        if item["definition"] is not None:
            continue
        user_input = f"Define the term '{item['vocab']}'."
        if provide_context or item.get("provideContext") and attachments:
            user_input = (
                f"your task: Define the target vocabulary below. "
                "Use the attached context file only as supporting context for the original text when it helps disambiguate the term. "
                f"Target item: '{item['vocab']}'"
            )
        if item.get("userPrompt"):
            user_input = "user prompt: " + item["userPrompt"].strip() + "\n\n" + user_input
        request = GeminiRequestConfig(
            model_name=model_name,
            system_instruction=(system_instruction),
            user_input=user_input,
            attachments=attachments if provide_context or item.get("provideContext") else [],
            generation=GenerationControls(),
        )
        result = executor.execute(request)
        item["definition"] = _normalize_generated_definition(result.text)
        processed += 1
        _print_generation_progress("Definition generation", processed, pending_total, started_at)
    return json_data

def get_footnote(
    json_data: list,
    model_name: str = "gemini-3.1-pro-preview",
    user_prompt: str = "",
    system_instruction: str = "",
    provide_context: bool = False,
    context_file_path: Path | None = None,
):
    executor = GeminiRequestExecutor()
    attachments = []
    if context_file_path is not None:
        attachments.append(
            AttachmentConfig(
                file_path=context_file_path,
                display_name=context_file_path.name,
            )
        )
    pending_total = sum(1 for item in json_data if item.get("footnote") is None)
    processed = 0
    started_at = time.monotonic()
    _print_generation_progress("Footnote generation", processed, pending_total, started_at)
    for item in json_data:
        if item.get("footnote") is not None:
            continue
        user_input = f"Target item: '{item['sentence']}'"
        if provide_context or item.get("provideContext") and attachments:
            user_input = "Use the attached context file only as supporting context for the original text when it helps disambiguate the sentence. " + '\n\n' + user_input
        if item.get("userPrompt"):
            user_input = "user_prompt: " + item["userPrompt"].strip() + "\n\n" + user_input
        request = GeminiRequestConfig(
            model_name=model_name,
            system_instruction=(system_instruction),
            user_input=user_input,
            attachments=attachments if provide_context or item.get("provideContext") else [],
            generation=GenerationControls(),
        )
        result = executor.execute(request)
        result_text = result.text.strip()
        print(f"Generated footnote for sentence '{item['sentence']}': {result_text}")
        item["footnote"] = result_text
        processed += 1
        _print_generation_progress("Footnote generation", processed, pending_total, started_at)
    return json_data

def write_definition_into_markdown(json_data: list[dict], output_path: Path):
    markdown_text = output_path.read_text(encoding="utf-8")
    definition_map: dict[str, str] = {}
    escaped_definition_map: dict[str, str] = {}
    vocab_patterns: list[str] = []
    definition_items = [
        (item["vocab"], item)
        for item in json_data
        if isinstance(item, dict) and item.get("definition")
    ]
    processed = 0
    started_at = time.monotonic()
    _print_generation_progress("Definition writing", processed, len(definition_items), started_at)

    for vocab, data in definition_items:
        definition = data["definition"]
        normalized_vocab = vocab.casefold()
        definition_map[normalized_vocab] = definition
        escaped_definition_map[normalized_vocab] = html.escape(definition, quote=True)
        vocab_patterns.append(re.escape(vocab))
        processed += 1
        _print_generation_progress("Definition writing", processed, len(definition_items), started_at)

    if not vocab_patterns:
        return

    vocab_pattern = "|".join(sorted(vocab_patterns, key=len, reverse=True))
    pattern = re.compile(
        r"<abbr\s+title=(['\"])(.*?)\1>("
        + vocab_pattern
        + r")</abbr>|(?<![\w-])("
        + vocab_pattern
        + r")(?![\w-])",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        existing_vocab = match.group(3)
        if existing_vocab is not None:
            normalized_vocab = existing_vocab.casefold()
            if html.unescape(match.group(2)) == definition_map[normalized_vocab]:
                return match.group(0)
            escaped_definition = escaped_definition_map[normalized_vocab]
            return f"<abbr title='{escaped_definition}'>{existing_vocab}</abbr>"

        matched_text = match.group(4)
        escaped_definition = escaped_definition_map[matched_text.casefold()]
        return f"<abbr title='{escaped_definition}'>{matched_text}</abbr>"

    output_path.write_text(pattern.sub(replace, markdown_text), encoding="utf-8")

def write_footnote_into_markdown(json_data: list[dict], output_path: Path):
    markdown_text = output_path.read_text(encoding="utf-8")
    abbr_tag_pattern = re.compile(r"</?abbr\b[^>]*>", flags=re.IGNORECASE)
    closing_abbr_pattern = re.compile(r"</abbr>", flags=re.IGNORECASE)
    footnote_pattern = re.compile(r"(?ms)^\[\^([^\]]+)\]:[ \t]*(.*?)(?=^\[\^[^\]]+\]:|\Z)")
    inline_marker_pattern = re.compile(r"\[\^([^\]]+)\]")
    pending_footnotes: dict[str, str] = {}
    footnote_items = [
        item for item in json_data
        if isinstance(item, dict)
        and str(item.get("sentence", "")).strip()
        and item.get("footnote")
    ]
    processed = 0
    started_at = time.monotonic()
    _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)

    def build_search_text(text: str) -> tuple[str, list[int]]:
        search_parts: list[str] = []
        index_map: list[int] = []
        last_end = 0
        for match in abbr_tag_pattern.finditer(text):
            if last_end < match.start():
                segment = text[last_end:match.start()]
                search_parts.append(segment)
                index_map.extend(range(last_end, match.start()))
            last_end = match.end()
        if last_end < len(text):
            search_parts.append(text[last_end:])
            index_map.extend(range(last_end, len(text)))
        return "".join(search_parts), index_map

    def collect_footnotes(text: str) -> dict[str, dict[str, object]]:
        footnotes: dict[str, dict[str, object]] = {}
        for match in footnote_pattern.finditer(text):
            footnotes[match.group(1)] = {
                "content": match.group(2).rstrip(),
                "span": match.span(),
            }
        return footnotes

    for item in footnote_items:
        sentence = str(item.get("sentence", "")).strip()
        footnote_text = str(item["footnote"]).strip()
        annotation_key = f"annotation_{hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:12]}"
        search_text, index_map = build_search_text(markdown_text)
        sentence_start = search_text.casefold().find(sentence.casefold())
        if sentence_start == -1:
            processed += 1
            _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)
            continue

        sentence_end = sentence_start + len(sentence)
        footnotes = collect_footnotes(markdown_text)
        existing_marker_match = inline_marker_pattern.match(search_text[sentence_end:])

        if existing_marker_match is not None:
            existing_key = existing_marker_match.group(1)
            existing_footnote = footnotes.get(existing_key)
            if existing_footnote and abbr_tag_pattern.sub("", existing_footnote["content"]) == footnote_text:
                processed += 1
                _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)
                continue

            if existing_footnote:
                start, end = existing_footnote["span"]
                markdown_text = markdown_text[:start] + markdown_text[end:]
                search_text, index_map = build_search_text(markdown_text)
                sentence_start = search_text.casefold().find(sentence.casefold())
                if sentence_start == -1:
                    processed += 1
                    _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)
                    continue
                sentence_end = sentence_start + len(sentence)
                existing_marker_match = inline_marker_pattern.match(search_text[sentence_end:])

            if existing_marker_match is not None:
                marker_text = existing_marker_match.group(0)
                marker_start = sentence_end
                original_start = index_map[marker_start]
                original_end = index_map[marker_start + len(marker_text) - 1] + 1
                markdown_text = (
                    markdown_text[:original_start]
                    + f"[^{annotation_key}]"
                    + markdown_text[original_end:]
                )
        else:
            insert_at = index_map[sentence_end - 1] + 1
            while True:
                closing_match = closing_abbr_pattern.match(markdown_text, insert_at)
                if closing_match is None:
                    break
                insert_at = closing_match.end()
            markdown_text = markdown_text[:insert_at] + f"[^{annotation_key}]" + markdown_text[insert_at:]

        footnotes = collect_footnotes(markdown_text)
        existing_annotation = footnotes.get(annotation_key)
        if existing_annotation and abbr_tag_pattern.sub("", existing_annotation["content"]) == footnote_text:
            processed += 1
            _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)
            continue
        if existing_annotation:
            start, end = existing_annotation["span"]
            markdown_text = markdown_text[:start] + markdown_text[end:]

        pending_footnotes[annotation_key] = footnote_text
        processed += 1
        _print_generation_progress("Footnote writing", processed, len(footnote_items), started_at)

    if pending_footnotes:
        footnote_lines = "\n\n".join(
            f"[^{annotation_key}]: {footnote_text}"
            for annotation_key, footnote_text in pending_footnotes.items()
        )
        markdown_text = markdown_text.rstrip() + "\n\n" + footnote_lines + "\n\n"

    output_path.write_text(markdown_text, encoding="utf-8")

def main(
        folder_path: Path, 
        context_file_stem: str = "original", 
        output_file_stem: str = "main", 
        old_vocab_file_stem: str = "vocab", 
        additional_vocab_file_stem: str = "vocab", 
        translated_sentences_file_stem: str = "translation", 
        additional_sentences_file_stem: str = "translation", 
        reset_definition_for_vocab_stem: str | None = "reset_vocab",
        reset_translation_for_sentence_stem: str | None = "reset_translation",
        system_instruction_for_definition_gen_path: Path = Path(__file__).parent / "definition_prompt.md", 
        system_instruction_for_translation_gen_path: Path = Path(__file__).parent / "translation_prompt.md",
        provide_context_for_definition_gen: bool = True,
        provide_context_for_translation_gen: bool = True,
        reset_all_existing_definitions: bool = False,
        reset_all_existing_translations: bool = False,    
        ):
    folder_path = Path(folder_path)

    def resolve_file(file_stem: str | Path, suffixes: tuple[str, ...]) -> Path:
        path = Path(file_stem)
        candidates: list[Path]
        if path.suffix:
            candidates = [path if path.is_absolute() else folder_path / path]
        else:
            base_path = path if path.is_absolute() else folder_path / path
            candidates = [base_path.with_suffix(suffix) for suffix in suffixes]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def resolve_instruction_path(path: Path) -> Path:
        return path if path.is_absolute() else folder_path / path

    context_file_path = resolve_file(context_file_stem, (".md", ".markdown"))
    output_text_path = resolve_file(output_file_stem, (".md", ".markdown"))
    old_vocab_list_path = resolve_file(old_vocab_file_stem, (".json",))
    additional_vocab_list_path = resolve_file(additional_vocab_file_stem, (".txt", ".md"))
    translated_sentences_list_path = resolve_file(translated_sentences_file_stem, (".json",))
    additional_sentences_list_path = resolve_file(additional_sentences_file_stem, (".txt", ".md"))
    reset_definition_list_path = resolve_file(reset_definition_for_vocab_stem, (".txt", ".md")) if reset_definition_for_vocab_stem else None
    reset_translation_list_path = resolve_file(reset_translation_for_sentence_stem, (".txt", ".md")) if reset_translation_for_sentence_stem else None
    system_instruction_for_definition_gen_path = resolve_instruction_path(system_instruction_for_definition_gen_path)
    system_instruction_for_translation_gen_path = resolve_instruction_path(system_instruction_for_translation_gen_path)

    if not output_text_path.exists():
        output_text_path.write_text(context_file_path.read_text(encoding="utf-8"), encoding="utf-8")

    system_instruction_for_definition_gen = system_instruction_for_definition_gen_path.read_text(encoding="utf-8")
    system_instruction_for_translation_gen = system_instruction_for_translation_gen_path.read_text(encoding="utf-8")

    json_data = add_new_vocab(old_vocab_list_path, additional_vocab_list_path)
    sentences_json_data = add_new_sentences(translated_sentences_list_path, additional_sentences_list_path)

    if reset_all_existing_definitions:
        json_data = reset_definitions(json_data)
    elif reset_definition_list_path is not None:
        json_data = reset_definitions(json_data, reset_definition_list_path)
        
    if reset_all_existing_translations:
        sentences_json_data = reset_footnote(sentences_json_data)
    elif reset_translation_list_path is not None:
        sentences_json_data = reset_footnote(sentences_json_data, reset_translation_list_path)

    json_data = get_definitions(
        json_data,
        system_instruction=system_instruction_for_definition_gen,
        context_file_path=context_file_path if provide_context_for_definition_gen else None,
    )
    sentences_json_data = get_footnote(
        sentences_json_data,
        system_instruction=system_instruction_for_translation_gen,
        context_file_path=context_file_path if provide_context_for_translation_gen else None,
    )

    old_vocab_list_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    translated_sentences_list_path.write_text(
        json.dumps(sentences_json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_footnote_into_markdown(sentences_json_data, output_text_path)
    write_definition_into_markdown(json_data, output_text_path)
    

if __name__ == "__main__":

    folder_path = input("Enter the folder path: ")
    folder_path = Path(rf"{folder_path.strip()}")

    main(
        folder_path,
        provide_context_for_definition_gen = False,
        provide_context_for_translation_gen = True,
        reset_all_existing_definitions = False,
        reset_all_existing_translations = False,
        )
