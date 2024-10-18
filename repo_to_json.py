import os

def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def create_repo_text(root_path):
    repo_structure = []

    for root, dirs, files in os.walk(root_path):
        # Exclude specific directories from being traversed
        if 'config' in dirs:
            dirs.remove('config')
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '.idea' in dirs:
            dirs.remove('.idea')
        if '.git' in dirs:
            dirs.remove('.git')
        # Special handling for 'terraform/.terraform' directory
        if 'terraform' in root and '.terraform' in dirs:
            dirs.remove('.terraform')

        # Special handling for 'terraform/.terraform' directory
        if 'webserver' in root and 'migrations' in dirs:
            dirs.remove('migrations')
        # Special handling for 'terraform/.terraform' directory
        if 'webserver' in root and '__pycache__' in dirs:
            dirs.remove('__pycache__')

        if 'terraform' in dirs:
            dirs.remove('terraform')
        if 'docs' in dirs:
            dirs.remove('docs')
        if 'unit_tests' in dirs:
            dirs.remove('unit_tests')

        # Generate relative path from the root path
        relative_path = os.path.relpath(root, root_path)
        repo_structure.append(f"Directory: {relative_path}" if relative_path != '.' else "Root Directory:")

        for file in files:
            # Skip certain file types (.yaml, .cfg, .md)
            if not file.endswith(('.cfg', '.md','.xml','.json','.lock','LICENSE','.terraform.lock.hcl', '__pycache__.py')):
                file_path = os.path.join(root, file)
                file_content = get_file_content(file_path)
                repo_structure.append(f"  File: {file}")
                repo_structure.append(f"  Content:\n{file_content}\n")

    # Save the plain text file in the root folder
    output_file = os.path.join(root_path, 'repo_structure.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(repo_structure))

    print(f"Repository structure saved to {output_file}")

# Usage
if __name__ == "__main__":
    # Use the current working directory (cwd)
    #repo_path = os.getcwd()
    # USe the folder src or another folder
    repo_path = os.path.join(os.getcwd(), 'webserver')

    create_repo_text(repo_path)