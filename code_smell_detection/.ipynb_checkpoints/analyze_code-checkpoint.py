import os
import subprocess


def _build_java_project(dir_path):
    print("Attempting compilation...")
    os.environ['JAVA_HOME']
    is_compiled = False
    pom_path = os.path.join(dir_path, 'pom.xml')
    orig_path = os.path.abspath(os.getcwd())
    if os.path.exists(pom_path):
        print("Found pom.xml")
        os.chdir(dir_path)
        proc = subprocess.Popen(
            [r'mvn', 'clean', 'install', '-DskipTests'])
        proc.wait()
        is_compiled = True
        os.chdir(orig_path)

    gradle_path = os.path.join(dir_path, "build.gradle")
    if os.path.exists(gradle_path):
        print("Found build.gradle")
        os.chdir(dir_path)
        try:
            proc = subprocess.Popen([r'gradle', 'compileJava'])
            proc.wait()
        except Exception as ex:
            print(ex)
            exit(1)
        is_compiled = True
        os.chdir(orig_path)
    if not is_compiled:
        print("Did not compile")


def _run_designite_java(folder_path, out_path, designite_path):
    print("Analyzing ...")
    proc = subprocess.Popen(
        ["java", "-jar", designite_path, "-i", folder_path, "-o", out_path, "-d"])
    proc.wait()
    print("done analyzing.")


def analyze_repo(folder_path, result_path, designite_path):
    print("Analyzing " + folder_path + " ...")

    _build_java_project(folder_path)
    _run_designite_java(folder_path, result_path, designite_path)


def get_type_dict(folder_path):
    cur_file = os.path.join(folder_path, 'TypeMetrics.csv')
    my_dict = dict()
    if os.path.isfile(cur_file):
        with open(cur_file, "r") as reader:
            is_header = True
            for line in reader:
                if is_header:
                    is_header = False
                    continue
                tokens = line.strip('\n').strip().split(',')
                if len(tokens) > 14:
                    key = tokens[0] + '.' + tokens[1] + '.' + tokens[2]
                    my_dict[key] = ",".join(tokens[0:14])
    return my_dict


def get_smell_dict(folder_path, smell_name):
    cur_file = os.path.join(folder_path, 'DesignSmells.csv')
    my_dict = dict()
    if os.path.isfile(cur_file):
        with open(cur_file, "r") as reader:
            is_header = True
            for line in reader:
                if is_header:
                    is_header = False
                    continue
                tokens = line.strip('\n').strip().split(',')
                if len(tokens) > 4:
                    if tokens[3].strip() == smell_name:
                        key = tokens[0] + '.' + tokens[1] + '.' + tokens[2]
                        my_dict[key] = ",".join(tokens[0:4])
    return my_dict


def create_dataset(type_dict, smell_dict, out_file):
    with open(out_file, 'a') as writer:
        for key in type_dict:
            smell_present = 1 if key in smell_dict else 0
            line = type_dict[key] + ',' + str(smell_present)
            writer.write(line + '\n')
