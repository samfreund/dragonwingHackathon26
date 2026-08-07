## Setting up the .venv

```bash
cd npu_qa
./run.ps1
cd ..
```

## Running the laptop router and laptop LLM and cloud LLM

```bash
npu_qa\.venv\Scripts\python.exe ask.py "How much RAM does the device have?" .\small_spec.txt
```

## Setting up the mobile device and IQ9

See the README under the `vlm-qa` and `mobile` folders.
