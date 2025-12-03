# Lightroom plugin installation
PLUGIN_NAME = NegativeAutoCrop.lrplugin
PLUGIN_DIR = $(HOME)/Library/Application Support/Adobe/Lightroom/Modules/$(PLUGIN_NAME)

.PHONY: init
init:
	@echo "Creating virtualenv..."
	@python3 -m venv venv
	@echo "Installing Python dependencies..."
	@./venv/bin/pip install -q -r requirements.txt
	@echo ""
	@echo "Virtualenv created. Activate with: source venv/bin/activate"

.PHONY: install
install:
	@mkdir -p "$(PLUGIN_DIR)"
	@mkdir -p "$(PLUGIN_DIR)/frame_detection"
	@echo "Installing Lua plugin files..."
	@for f in plugin/*.lua; do \
		cp "$$f" "$(PLUGIN_DIR)/"; \
		echo "  Installed $$(basename $$f)"; \
	done
	@echo "Installing Python package..."
	@for f in frame_detection/*.py; do \
		cp "$$f" "$(PLUGIN_DIR)/frame_detection/"; \
		echo "  Installed frame_detection/$$(basename $$f)"; \
	done
	@cp requirements.txt "$(PLUGIN_DIR)/"
	@echo "Creating virtualenv..."
	@python3 -m venv "$(PLUGIN_DIR)/venv"
	@echo "Installing Python dependencies..."
	@"$(PLUGIN_DIR)/venv/bin/pip" install -q -r "$(PLUGIN_DIR)/requirements.txt"
	@echo ""
	@echo "Plugin installed to $(PLUGIN_DIR)"

.PHONY: uninstall
uninstall:
	@rm -rf "$(PLUGIN_DIR)"
	@echo "Plugin uninstalled"

.PHONY: clean
clean:
	@rm -rf __pycache__ frame_detection/__pycache__
	@rm -rf debug/*.jpg debug/*.png
	@echo "Cleaned build artifacts"

# Run frame detection on a test image (for development)
.PHONY: test
test:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make test IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	./venv/bin/python -m frame_detection "$(IMAGE)" --debug-dir debug
