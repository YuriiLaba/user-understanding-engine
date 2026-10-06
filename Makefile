PYTHON := python3.12
VENV := .dev_env
ACTIVATE := $(VENV)/bin/activate
REQUIREMENTS := requirements.txt

.PHONY: all setup install clean

all: setup install

setup:
	@echo "Creating virtual environment in $(VENV)..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi

install: $(REQUIREMENTS)
	@echo "Installing Python dependencies..."
	@. $(ACTIVATE) && pip install --upgrade pip && pip install -r $(REQUIREMENTS)
	@echo "Dependencies installed."

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Cleaned up."