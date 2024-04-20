import git
import os


def get_git_root():
    cwd = os.getcwd()
    git_repo = git.Repo(cwd, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root
