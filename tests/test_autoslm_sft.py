import json

import pytest

from autoslm_v3.sft import SFTConfig, _extract_prompt_answer, config_from_json, load_jsonl, prepare_splits


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
