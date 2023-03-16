# Code Search

### Task: Code Search using CoSQA dataset

- Dataset: CodeXGlue [Link](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery/data)
- Dataset will be download from S3 bucket using the download and preprocess data method shell script file
- Downstream Task: Code Search
- Downstream Description: Based on natural language search provide code snippets as output
- Usage: To search for codes/ Stack Overflow
- File: train.ipynb
- Models: outputs

**Sample Data:**

```
{
      "idx": "webquery-test-1",
      "doc": "how to open a text file on python",
      "code": "def get_file_string(filepath):     \"\"\"Get string from file.\"\"\"     with  
               open(os.path.abspath(filepath)) as f:         return f.read()"
}
```
 
<br/>

#### How to run:
1. Open train.ipynb
2. Play around with the argument settings to change the language models, change the type of tokenizer etc.
3. create the custom model
4. Train the model
5. Evaluate the model

##### Note: Reach out to me on teams in case if you face any issues with the execution.

