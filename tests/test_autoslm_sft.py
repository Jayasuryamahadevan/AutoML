import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoslm_v3.sft import (SFTConfig, _extract_prompt_answer, _normalized_answer, _token_f1, config_from_json, load_jsonl, prepare_splits)


def test_rtx3050_preset_is_valid():
    config = SFTConfig()
    config.validate()
    assert config.max_seq_length == 512
    assert config.per_device_train_batch_size == 1
    assert config.lora_rank == 8
    assert config.use_4bit


def test_config_rejects_invalid_low_vram_batch_size():
    with pytest.raises(ValueError):
        SFTConfig(per_device_train_batch_size=2).validate()


def test_config_rejects_invalid_split():
    with pytest.raises(ValueError):
        SFTConfig(test_fraction=0.6, validation_fraction=0.5).validate()


def test_extract_instruction_input_response():
    prompt, answer = _extract_prompt_answer({"instruction": "Add", "input": "2 + 2", "response": "4"})
    assert "2 + 2" in prompt
    assert answer == "4"


def test_extract_prompt_completion_aliases():
    prompt, answer = _extract_prompt_answer({"prompt": "Question", "completion": "Answer"})
    assert prompt == "Question"
    assert answer == "Answer"


def test_messages_format():
    prompt, answer = _extract_prompt_answer({"messages": [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]})
    assert "2+2?" in prompt
    assert answer == "4"


def test_jsonl_validation_and_load(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({
        "instruction": f"Question {i}", "response": f"Answer {i}", "group_id": f"doc-{i}"
    }) for i in range(20)), encoding="utf-8")
    rows = load_jsonl(str(path))
    split = prepare_splits(rows, SFTConfig())
    assert split.train and split.validation and split.test


def test_invalid_jsonl_fails_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction":"a","response":"b"}\n{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(str(path))


def test_config_loader(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"target_modules": ["q_proj"], "max_seq_length": 256}), encoding="utf-8")
    config = config_from_json(str(path))
    assert config.target_modules == ("q_proj",)
    assert config.max_seq_length == 256


def test_normalized_answer_collapses_whitespace():
    assert _normalized_answer(" A  B\nC ") == "a b c"


def test_token_f1_is_bounded_and_exact_match_is_one():
    assert _token_f1("the quick fox", "the quick fox") == 1.0
    assert 0.0 < _token_f1("the quick fox", "the slow fox") < 1.0
    assert _token_f1("abc", "xyz") == 0.0


def test_repository_cli_help_runs_without_ml_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_sft.py"), "--help"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "--resume-from-checkpoint" in result.stdout

 
def test_messages_format_preserves_chat_roles():
    from autoslm_v3.sft import _extract_prompt_answer, _format_row_prompt
    row={"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Explain LoRA."},{"role":"assistant","content":"Low-rank adaptation."}]}
    prompt,answer=_extract_prompt_answer(row)
    assert "system: Be concise." in prompt and "user: Explain LoRA." in prompt
    assert answer=="Low-rank adaptation."
    class ChatTokenizer:
        chat_template="template"
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            assert messages==row["messages"][:-1]
            assert tokenize is False and add_generation_prompt is True
            return "ROLE_PRESERVING_PROMPT"
    assert _format_row_prompt(ChatTokenizer(),{**row,"instruction":prompt})=="ROLE_PRESERVING_PROMPT"


def test_messages_format_rejects_non_final_assistant_turn():
    from autoslm_v3.sft import _extract_prompt_answer
    row={"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"Again"},{"role":"assistant","content":"Yes"}]}
    with pytest.raises(ValueError,match="single final assistant"):
        _extract_prompt_answer(row)
