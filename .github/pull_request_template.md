## Summary

<!-- What changed and why? -->

## Risk area

<!-- Check all that apply. -->

- [ ] Model files / checkpoints
- [ ] Download URLs or mirrors
- [ ] HuggingFace / ModelScope cache layout
- [ ] Example audio files
- [ ] Inference behavior
- [ ] Dependencies / packaging
- [ ] Docs only

## Regression checklist

- [ ] Pulling this change will not delete user-downloaded model files.
- [ ] Missing small files, such as `config.yaml`, do not trigger a full model re-download.
- [ ] Existing auxiliary models in `checkpoints/hf_cache` can still be reused.
- [ ] Existing auxiliary models in `~/.cache/huggingface/hub` can still be reused.
- [ ] Download fallback paths are intentional and do not silently hide invalid URLs.

## Validation

<!-- List commands run, or explain why validation is not needed. -->

- [ ] `python -m compileall -q indextts webui.py`
- [ ] `python -m unittest discover -s tests -p "test_*.py"`
