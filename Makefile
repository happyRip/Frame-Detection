.PHONY: init
init:
	@echo "Creating virtualenv..."
	@python3 -m venv venv
	@echo "Installing package in editable mode..."
	@./venv/bin/pip install -q -e .
	@echo ""
	@echo "Virtualenv created. Activate with: source venv/bin/activate"
	@echo "The 'negative-auto-crop' command is now available in the venv."

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
