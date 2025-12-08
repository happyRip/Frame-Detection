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
	@python3 installer.py install

.PHONY: uninstall
uninstall:
	@python3 installer.py uninstall

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
