import os
import subprocess
import shutil


# To figure out whether a file contains multiple definitions of a method,
# we tokenize the file using "method" config and checks whether it contains multiple lines.
def _is_contain_overloaded_methods(input_file, tokenizer_path, tokenizer_language):
    temp_file = "temp.tok"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    with open(temp_file, "w", errors='ignore') as tok_out_file:
        process = subprocess.Popen([tokenizer_path, "-l", tokenizer_language, "-o", "method", input_file],
                                   bufsize=10240000, stdout=tok_out_file)
        process.wait()

    with open(temp_file, "r", errors='ignore') as in_f:
        tok_text = in_f.read()
    length = 0
    for line in tok_text.lstrip("b'").rstrip("'\\n").split("\n"):
        if len(line) > 0:
            length += 1
    if(length > 1):
        return True
    return False


def run_tokenizer(folder_path, out_folder, tokenizer_path, tokenizer_language, tokenizer_level):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    file_counter = 1
    out_file = os.path.join(out_folder, "tokenized" + str(file_counter) + ".tok")
    for file in os.listdir(folder_path):
        input_file = os.path.abspath(os.path.join(folder_path, file))
        try:
            print("\t\tprocessing " + file)
        except:
            pass
        in_file = os.path.abspath(os.path.join(out_folder, "temp.tok"))
        if os.path.exists(in_file):
            os.remove(in_file)
        with open(in_file, "w", errors='ignore') as tok_out_file:
            exe_path = os.path.abspath(tokenizer_path)
            # process = subprocess.Popen([exe_path, "-l", tokenizer_language, "-o",
            #                             tokenizer_level, input_file],
            #                            bufsize=10240000, stdout=tok_out_file, shell=True)
            subprocess.run([exe_path, "-l", tokenizer_language, "-o",
                                        tokenizer_level, input_file], stdout=tok_out_file)
            # process.wait()

        with open(out_file, "a", errors='ignore') as f:
            with open(in_file, "r", errors='ignore') as in_f:
                tok_text = in_f.read()
            f.write(tok_text.lstrip("b'").rstrip("'\\n"))
        os.remove(in_file)
        if os.path.getsize(out_file) > 52428800: #50 mb
            file_counter += 1
            out_file = os.path.join(out_folder, "tokenized" + str(file_counter) + ".tok")


def _get_max_length(tok_text):
    max_length = 0
    for line in tok_text.split('\n'):
        tokens = line.split('\t')
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length


