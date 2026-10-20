#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Remove Claude attribution from Git history
# Repository:
# https://github.com/csarvind2000/MuscleSegmentation
# ============================================================

ARVIND_NAME="Arvind C S"
ARVIND_EMAIL="csarvind2000@gmail.com"

REPO_URL="https://github.com/csarvind2000/MuscleSegmentation.git"
BRANCH="main"

echo "============================================================"
echo " Remove Claude attribution from MuscleSegmentation"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Check that this is a Git repository
# ------------------------------------------------------------
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: You are not inside a Git repository."
    echo
    echo "Go to your repository first, for example:"
    echo "cd /media/ranjhaa-local/volume23/sarcopenia/seg_benchmark"
    exit 1
fi

# ------------------------------------------------------------
# 2. Show repository location
# ------------------------------------------------------------
REPO_DIR="$(git rev-parse --show-toplevel)"

echo "Repository:"
echo "$REPO_DIR"
echo

cd "$REPO_DIR"

# ------------------------------------------------------------
# 3. Ensure the working tree is clean
# ------------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Your repository has uncommitted changes."
    echo
    git status
    echo
    echo "Commit or stash your changes before running this script."
    exit 1
fi

# ------------------------------------------------------------
# 4. Configure your Git identity
# ------------------------------------------------------------
echo "Setting Git identity..."

git config user.name "$ARVIND_NAME"
git config user.email "$ARVIND_EMAIL"

echo
echo "Git identity:"
echo "Name : $(git config user.name)"
echo "Email: $(git config user.email)"
echo

# ------------------------------------------------------------
# 5. Check/install git-filter-repo
# ------------------------------------------------------------
if ! command -v git-filter-repo >/dev/null 2>&1; then

    echo "git-filter-repo is not installed."
    echo "Attempting installation..."
    echo

    if command -v pip3 >/dev/null 2>&1; then
        pip3 install --user git-filter-repo
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m pip install --user git-filter-repo
    else
        echo "ERROR: Python/pip is not available."
        echo
        echo "Install git-filter-repo manually:"
        echo "python3 -m pip install --user git-filter-repo"
        exit 1
    fi

    # Add typical user binary directory to PATH
    export PATH="$HOME/.local/bin:$PATH"

fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
    echo
    echo "ERROR: git-filter-repo installation could not be detected."
    echo
    echo "Try manually:"
    echo "python3 -m pip install --user git-filter-repo"
    echo
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    exit 1
fi

echo "git-filter-repo found:"
git-filter-repo --version
echo

# ------------------------------------------------------------
# 6. Fetch latest repository state
# ------------------------------------------------------------
echo "Fetching latest GitHub state..."

if git remote get-url origin >/dev/null 2>&1; then
    git fetch origin --prune
else
    echo "No origin remote found."
    echo "Adding:"
    echo "$REPO_URL"
    git remote add origin "$REPO_URL"
    git fetch origin
fi

echo

# ------------------------------------------------------------
# 7. Show Claude references BEFORE rewriting
# ------------------------------------------------------------
echo "============================================================"
echo " Claude references BEFORE cleanup"
echo "============================================================"
echo

FOUND_BEFORE=0

while IFS= read -r line; do
    echo "$line"
    FOUND_BEFORE=1
done < <(
    git log --all \
        --pretty=format:'Commit: %H%nAuthor: %an <%ae>%nCommitter: %cn <%ce>%nSubject: %s%n%b%n---------------------------' \
        | grep -i -B5 -A5 "claude" || true
)

if [[ "$FOUND_BEFORE" -eq 0 ]]; then
    echo "No explicit 'Claude' text found in commit history."
    echo
    echo "Claude may still appear because of commit email attribution."
    echo "The script will continue and clean Claude-related author metadata."
fi

echo

# ------------------------------------------------------------
# 8. Create full Git backup
# ------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_PARENT="$(dirname "$REPO_DIR")"
BACKUP_FILE="$BACKUP_PARENT/MuscleSegmentation-before-claude-cleanup-$TIMESTAMP.bundle"

echo "Creating Git backup..."
echo
echo "$BACKUP_FILE"
echo

git bundle create "$BACKUP_FILE" --all

echo "Backup successfully created."
echo

# ------------------------------------------------------------
# 9. Rewrite Git history
# ------------------------------------------------------------
echo "============================================================"
echo " Rewriting Git history"
echo "============================================================"
echo

git filter-repo --force \
    --commit-callback "
# ------------------------------------------------------------
# Replace Claude as AUTHOR
# ------------------------------------------------------------

author_name = commit.author_name.lower()
author_email = commit.author_email.lower()

if b'claude' in author_name or b'claude' in author_email:
    commit.author_name = b'$ARVIND_NAME'
    commit.author_email = b'$ARVIND_EMAIL'


# ------------------------------------------------------------
# Replace Claude as COMMITTER
# ------------------------------------------------------------

committer_name = commit.committer_name.lower()
committer_email = commit.committer_email.lower()

if b'claude' in committer_name or b'claude' in committer_email:
    commit.committer_name = b'$ARVIND_NAME'
    commit.committer_email = b'$ARVIND_EMAIL'


# ------------------------------------------------------------
# Remove Claude Co-authored-by lines
# ------------------------------------------------------------

message_lines = commit.message.splitlines()

clean_lines = []

for line in message_lines:

    lower_line = line.lower()

    # Remove lines such as:
    #
    # Co-authored-by: Claude <...>
    #
    if b'co-authored-by:' in lower_line and b'claude' in lower_line:
        continue

    # Remove generated-with-Claude attribution
    if b'claude' in lower_line and (
        b'co-authored' in lower_line
        or b'generated with' in lower_line
        or b'generated by' in lower_line
        or b'assisted by' in lower_line
    ):
        continue

    clean_lines.append(line)


commit.message = b'\n'.join(clean_lines)

if commit.message and not commit.message.endswith(b'\n'):
    commit.message += b'\n'
"

echo
echo "History rewrite finished."
echo

# ------------------------------------------------------------
# 10. git-filter-repo removes origin intentionally
# ------------------------------------------------------------
echo "Checking remote..."

if ! git remote get-url origin >/dev/null 2>&1; then

    echo "Restoring origin remote..."

    git remote add origin "$REPO_URL"

fi

echo
echo "Origin:"
git remote -v
echo

# ------------------------------------------------------------
# 11. Check history AFTER cleanup
# ------------------------------------------------------------
echo "============================================================"
echo " Checking history AFTER cleanup"
echo "============================================================"
echo

CLAUDE_RESULTS="$(
    git log --all \
        --pretty=format:'%H | %an <%ae> | %cn <%ce> | %s%n%b' \
        | grep -i "claude" || true
)"

if [[ -n "$CLAUDE_RESULTS" ]]; then

    echo "WARNING:"
    echo "Some references containing the word Claude still exist:"
    echo
    echo "$CLAUDE_RESULTS"
    echo
    echo "Review these before force pushing."

else

    echo "SUCCESS:"
    echo "No Claude attribution was found in rewritten Git history."

fi

echo

# ------------------------------------------------------------
# 12. Show rewritten commits
# ------------------------------------------------------------
echo "============================================================"
echo " Current commits"
echo "============================================================"
echo

git log \
    --pretty=format:'%h | %an <%ae> | %s' \
    --decorate \
    --max-count=20

echo
echo

# ------------------------------------------------------------
# 13. Confirmation before destructive force push
# ------------------------------------------------------------
echo "============================================================"
echo " IMPORTANT"
echo "============================================================"
echo
echo "The local history has been rewritten."
echo
echo "The following command will replace the GitHub branch history:"
echo
echo "git push origin $BRANCH --force"
echo
echo "Backup:"
echo "$BACKUP_FILE"
echo

read -r -p "Force push cleaned history to GitHub? [y/N]: " ANSWER

if [[ ! "$ANSWER" =~ ^[Yy]$ ]]; then

    echo
    echo "Force push cancelled."
    echo
    echo "Nothing has been changed on GitHub yet."
    echo
    echo "Your local repository has rewritten history."
    echo
    echo "To push manually later:"
    echo
    echo "git push origin $BRANCH --force"
    echo
    exit 0

fi

# ------------------------------------------------------------
# 14. Push rewritten main branch
# ------------------------------------------------------------
echo
echo "Force pushing cleaned history..."

git push origin "$BRANCH" --force

echo
echo "============================================================"
echo " COMPLETED"
echo "============================================================"
echo
echo "Repository:"
echo "https://github.com/csarvind2000/MuscleSegmentation"
echo
echo "Claude attribution has been removed from the rewritten"
echo "Git commit history and the main branch has been pushed."
echo
echo "Backup:"
echo "$BACKUP_FILE"
echo
echo "NOTE:"
echo "GitHub's Contributors panel may take some time to recalculate."
echo "Refresh the repository after a few minutes."
echo