_Release Notes: v0.0.7 -> 8f760ae36c67f35422fef3d6a4233000f81db178_

-----

### :white_check_mark: Bug Fixes
* [(6aa148a)](https://github.com/google/ml-flashpoint/+/6aa148a430bc2033b28bf62d56038c0eb025f6d7) adapter/nemo: Fix buffer pool init when initial_write_buffer_size_bytes is None (#80)

### :clock1: Performance
* [(17457aa)](https://github.com/google/ml-flashpoint/+/17457aae7fea1ba6efbe317a1fa88077bb2d3891) adapter/nemo: reduce NUM_OF_BUFFERS_PER_OBJECT to 2 to reduce memory pressure (#81)
* [(71217a4)](https://github.com/google/ml-flashpoint/+/71217a45a143b26c281588f9211f76e5821270e5) Implement BufferPool for efficient memory reuse (#61)

### :arrows_clockwise: CI
* [(8f760ae)](https://github.com/google/ml-flashpoint/+/8f760ae36c67f35422fef3d6a4233000f81db178) fix VERSION extraction from TAG_NAME in cloudbuild.yaml (#85)
* [(cba3c3a)](https://github.com/google/ml-flashpoint/+/cba3c3af268aa79388bf27b70146a0e5a9638b93) fix $$VERSION env var syntax (#82)
* [(75bf47a)](https://github.com/google/ml-flashpoint/+/75bf47a4bb2160cf9b567073832ce90272db23dc) validate TAG_NAME and force the version used for pypi upload (#79)

-----

_Generated with: `./scripts/create_release.py`_