```markdown
# vitfly Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and coding conventions used in the `vitfly` Python codebase. You'll learn how to structure files, write imports and exports, follow commit message conventions, and understand the project's approach to testing. This guide is ideal for contributors who want to maintain consistency and quality in their work.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `data_loader.py`, `model_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import preprocess_data
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ['DataLoader', 'preprocess_data']
    ```

### Commit Messages
- Follow **conventional commit** style.
- Use the `fix` prefix for bug fixes.
- Keep commit messages concise (average ~55 characters).
  - Example:
    ```
    fix: handle edge case in data preprocessing
    ```

## Workflows

### Fixing a Bug
**Trigger:** When you identify and resolve a bug in the codebase  
**Command:** `/fix-bug`

1. Locate and fix the bug in the relevant Python file.
2. Write a test (or update an existing one) in a `*.test.*` file to cover the fix.
3. Commit your changes using the `fix:` prefix.
   - Example: `fix: correct off-by-one error in batch generator`
4. Push your changes and open a pull request.

### Adding a New Module
**Trigger:** When you need to add a new feature or module  
**Command:** `/add-module`

1. Create a new Python file using snake_case naming.
2. Implement your feature using relative imports as needed.
3. Add named exports via `__all__`.
4. Write corresponding tests in a `*.test.*` file.
5. Commit your changes with a descriptive message.
   - Example: `feat: add data augmentation module`

### Writing Tests
**Trigger:** When adding or updating code that requires test coverage  
**Command:** `/write-test`

1. Create or update a test file matching the pattern `*.test.*`.
   - Example: `data_loader.test.py`
2. Implement tests for your functions or classes.
3. Run the tests using your preferred Python test runner.
4. Commit test changes with a relevant message.
   - Example: `test: add edge case tests for data loader`

## Testing Patterns

- Test files follow the pattern `*.test.*` (e.g., `module.test.py`).
- The testing framework is not specified; use standard Python testing tools such as `unittest` or `pytest`.
- Place tests alongside the code or in a dedicated tests directory.
- Example test structure:
  ```python
  import unittest
  from .data_loader import DataLoader

  class TestDataLoader(unittest.TestCase):
      def test_loads_data(self):
          loader = DataLoader()
          self.assertIsNotNone(loader.load())
  ```

## Commands
| Command      | Purpose                                      |
|--------------|----------------------------------------------|
| /fix-bug     | Guide for fixing and committing a bug         |
| /add-module  | Steps for adding a new module or feature      |
| /write-test  | Instructions for writing and committing tests |
```
