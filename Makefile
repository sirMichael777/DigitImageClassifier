.PHONY: all clean run

# Target executable name
EXECUTABLE := Classifier

# Source directory
SRC_DIR := src

# Source files
SRC := $(wildcard $(SRC_DIR)/*.py)

# Compilation command (not applicable for Python)
# Use this space for any compilation flags or other setup

# Specify the path to the Python interpreter
PYTHON := /usr/bin/python3

# Default target
all: $(EXECUTABLE)

# Rule to build the executable
$(EXECUTABLE):

# Rule to run the program
run: $(EXECUTABLE)
	$(PYTHON) $(SRC)

# Rule to clean generated files
clean:
	@rm -rf $(EXECUTABLE)
