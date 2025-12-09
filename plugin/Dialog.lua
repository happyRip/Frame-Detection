-- Settings dialog for AutoCrop plugin

local LrApplication = import("LrApplication")
local LrBinding = import("LrBinding")
local LrDialogs = import("LrDialogs")
local LrFileUtils = import("LrFileUtils")
local LrFunctionContext = import("LrFunctionContext")
local LrPathUtils = import("LrPathUtils")
local LrShell = import("LrShell")
local LrTasks = import("LrTasks")
local LrView = import("LrView")

local AutoCrop = require("AutoCrop")
local Info = require("Info")
local Settings = require("Settings")

-- Version string from Info.lua
local VERSION_STRING = string.format("%d.%d.%d", Info.VERSION.major, Info.VERSION.minor, Info.VERSION.revision)

local ASPECT_RATIOS = {
	{ title = "1:1", value = "1:1" },
	{ title = "2:3", value = "2:3" },
	{ title = "3:4", value = "3:4" },
	{ title = "4:5", value = "4:5" },
	{ title = "6:4.5", value = "6:4.5" },
	{ title = "6:6", value = "6:6" },
	{ title = "6:7", value = "6:7" },
	{ title = "6:9", value = "6:9" },
	{ title = "Custom", value = "custom" },
}

local FILM_TYPES = {
	{ title = "Auto-detect", value = "auto" },
	{ title = "Negative", value = "negative" },
	{ title = "Positive", value = "positive" },
}

local EDGE_FILTERS = {
	{ title = "Canny", value = "canny" },
	{ title = "Sobel", value = "sobel" },
	{ title = "Scharr (default)", value = "scharr" },
	{ title = "DoG (Difference of Gaussians)", value = "dog" },
	{ title = "Laplacian", value = "laplacian" },
	{ title = "LoG (Laplacian of Gaussian)", value = "log" },
}

local SEPARATION_METHODS = {
	{ title = "Color Distance (default)", value = "color_distance" },
	{ title = "CLAHE", value = "clahe" },
	{ title = "LAB Distance", value = "lab_distance" },
	{ title = "HSV Distance", value = "hsv_distance" },
	{ title = "Adaptive", value = "adaptive" },
	{ title = "Gradient", value = "gradient" },
}

--------------------------------------------------------------------------------
-- UI Constants
--------------------------------------------------------------------------------

local LABEL_WIDTH = 100
local POPUP_WIDTH = 150
local PATH_WIDTH = 300
local DIGITS_WIDTH = 4
local CHARS_WIDTH = 12
local SLIDER_WIDTH = 150

--------------------------------------------------------------------------------
-- UI Tabs
--------------------------------------------------------------------------------

local function buildCropSettingsTab(f, props, restoreDefaults, runAutoCrop, navigatePrev, navigateNext)
	return f:tab_view_item({
		title = "Crop Settings",
		identifier = "crop",
		f:row({
			fill_horizontal = 1,
			f:column({
				bind_to_object = props,
				spacing = f:control_spacing(),
				place = "horizontal_center",
				fill_horizontal = 1,

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Aspect ratio", width = LABEL_WIDTH, alignment = "right" }),
					f:popup_menu({
						items = ASPECT_RATIOS,
						value = LrView.bind("aspectRatio"),
						width = POPUP_WIDTH,
						tooltip = "Select the target aspect ratio for the cropped frame.",
					}),
					f:edit_field({
						value = LrView.bind("customAspectRatio"),
						width_in_chars = 8,
						visible = LrView.bind({
							key = "aspectRatio",
							transform = function(value)
								return value == "custom"
							end,
						}),
						tooltip = "Enter a custom aspect ratio (e.g., 16:9).",
					}),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Film type", width = LABEL_WIDTH, alignment = "right" }),
					f:popup_menu({
						items = FILM_TYPES,
						value = LrView.bind("filmType"),
						width = POPUP_WIDTH,
						tooltip = "Select the film type. Negative film has bright sprocket holes, positive (slide) film has dark sprocket holes. Auto-detect will try to determine the type automatically.",
					}),
				}),

				f:row({
					f:static_text({ title = "Crop in", width = LABEL_WIDTH, alignment = "right" }),
					f:edit_field({
						value = LrView.bind("cropIn"),
						width_in_digits = DIGITS_WIDTH,
						precision = 1,
						min = 0,
						max = 15,
						alignment = "right",
						tooltip = "Sets the maximum percentage to crop inward from detected edges. This helps account for rounded corners while minimizing the crop when possible.",
					}),
					f:static_text({ title = "%" }),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Sprocket margin", width = LABEL_WIDTH, alignment = "right" }),
					f:edit_field({
						value = LrView.bind("sprocketMargin"),
						width_in_digits = DIGITS_WIDTH,
						precision = 2,
						min = 0,
						max = 10,
						alignment = "right",
						tooltip = "Percentage to crop beyond detected sprocket holes (0-10).",
					}),
					f:static_text({ title = "%" }),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Film base inset", width = LABEL_WIDTH, alignment = "right" }),
					f:edit_field({
						value = LrView.bind("filmBaseInset"),
						width_in_digits = DIGITS_WIDTH,
						precision = 2,
						min = 0,
						max = 50,
						alignment = "right",
						tooltip = "Diagonal inset percentage for film base sampling region (used when no sprocket holes detected).",
					}),
					f:static_text({ title = "%" }),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Edge margin", width = LABEL_WIDTH, alignment = "right" }),
					f:edit_field({
						value = LrView.bind("edgeMargin"),
						width_in_chars = CHARS_WIDTH,
						tooltip = "Defines the percentage of image edges to search for frame boundaries. Use a single value (e.g., 5), two values for vertical and horizontal (e.g., 5,10), or four values for each edge (e.g., 5,10,7.5,13).",
					}),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:static_text({ title = "Ignore margin", width = LABEL_WIDTH, alignment = "right" }),
					f:edit_field({
						value = LrView.bind("ignoreMargin"),
						width_in_chars = CHARS_WIDTH,
						tooltip = "Specifies the percentage of image margins to exclude from analysis. Use a single value (e.g., 5), two values for vertical and horizontal (e.g., 0,5), or four values for each edge (e.g., 0,5,0,5).",
					}),
				}),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:checkbox({
						title = "Reset crop",
						value = LrView.bind("resetCrop"),
						tooltip = "Resets any existing crop before processing the image.",
					}),
					f:push_button({
						title = "Restore Defaults",
						action = restoreDefaults,
					}),
				}),

				f:spacer({ height = 10 }),

				f:row({
					fill_horizontal = 1,
					alignment = "center",
					f:push_button({
						title = "<",
						action = navigatePrev,
						width = 30,
						tooltip = "Go to previous photo",
					}),
					f:push_button({
						title = "Auto Crop",
						action = runAutoCrop,
						tooltip = "Run auto crop on selected photos",
					}),
					f:push_button({
						title = ">",
						action = navigateNext,
						width = 30,
						tooltip = "Go to next photo",
					}),
				}),
			}),
		}),
	})
end

local function buildFiltersTab(f, props, generatePreview)
	-- Helper function for conditional visibility based on edge filter selection
	local function visibleForFilter(filterValue)
		return LrView.bind({
			key = "edgeFilter",
			transform = function(value)
				return value == filterValue
			end,
		})
	end

	-- Helper function for showing blur_size filters (sobel, scharr, laplacian)
	local function visibleForBlurFilters()
		return LrView.bind({
			key = "edgeFilter",
			transform = function(value)
				return value == "sobel" or value == "scharr" or value == "laplacian"
			end,
		})
	end

	-- Helper function for conditional visibility based on separation method
	local function visibleForSeparation(methodValue)
		return LrView.bind({
			key = "separationMethod",
			transform = function(value)
				return value == methodValue
			end,
		})
	end

	return f:tab_view_item({
		title = "Filters",
		identifier = "filters",
		f:column({
			bind_to_object = props,
			spacing = f:control_spacing(),
			fill_horizontal = 1,

			-- Edge Filter Section
			f:static_text({
				title = "Edge Detection",
				font = "<system/bold>",
			}),

			f:row({
				f:static_text({ title = "Edge filter", width = LABEL_WIDTH, alignment = "right" }),
				f:popup_menu({
					items = EDGE_FILTERS,
					value = LrView.bind("edgeFilter"),
					width = POPUP_WIDTH,
					tooltip = "Select the edge detection filter for frame boundary detection. Scharr is recommended for most cases.",
				}),
			}),

			-- Canny parameters
			f:row({
				visible = visibleForFilter("canny"),
				f:static_text({ title = "Low threshold", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("cannyLow"),
					min = 0,
					max = 255,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("cannyLow"),
					width_in_digits = 3,
					min = 0,
					max = 255,
					tooltip = "Lower threshold for Canny edge detection (0-255). Lower values detect more edges.",
				}),
			}),
			f:row({
				visible = visibleForFilter("canny"),
				f:static_text({ title = "High threshold", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("cannyHigh"),
					min = 0,
					max = 255,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("cannyHigh"),
					width_in_digits = 3,
					min = 0,
					max = 255,
					tooltip = "Upper threshold for Canny edge detection (0-255). Higher values filter weaker edges.",
				}),
			}),

			-- Blur size for Sobel/Scharr/Laplacian
			f:row({
				visible = visibleForBlurFilters(),
				f:static_text({ title = "Blur size", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind({
						keys = { "edgeFilter", "sobelBlurSize", "scharrBlurSize", "laplacianBlurSize" },
						operation = function(binder, values)
							local filter = values.edgeFilter
							if filter == "sobel" then
								return values.sobelBlurSize
							elseif filter == "scharr" then
								return values.scharrBlurSize
							elseif filter == "laplacian" then
								return values.laplacianBlurSize
							end
							return 5
						end,
					}),
					min = 0,
					max = 21,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind({
						keys = { "edgeFilter", "sobelBlurSize", "scharrBlurSize", "laplacianBlurSize" },
						operation = function(binder, values)
							local filter = values.edgeFilter
							if filter == "sobel" then
								return values.sobelBlurSize
							elseif filter == "scharr" then
								return values.scharrBlurSize
							elseif filter == "laplacian" then
								return values.laplacianBlurSize
							end
							return 5
						end,
					}),
					width_in_digits = 2,
					min = 0,
					max = 21,
					tooltip = "Gaussian blur kernel size (0 or odd numbers 1-21). Higher values reduce noise but may blur edges.",
				}),
			}),

			-- DoG parameters
			f:row({
				visible = visibleForFilter("dog"),
				f:static_text({ title = "Sigma 1", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("dogSigma1"),
					min = 0.1,
					max = 5.0,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("dogSigma1"),
					width_in_digits = 4,
					precision = 1,
					min = 0.1,
					max = 5.0,
					tooltip = "First Gaussian sigma for DoG filter (0.1-5.0). Controls fine detail level.",
				}),
			}),
			f:row({
				visible = visibleForFilter("dog"),
				f:static_text({ title = "Sigma 2", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("dogSigma2"),
					min = 0.1,
					max = 10.0,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("dogSigma2"),
					width_in_digits = 4,
					precision = 1,
					min = 0.1,
					max = 10.0,
					tooltip = "Second Gaussian sigma for DoG filter (0.1-10.0). Should be larger than Sigma 1.",
				}),
			}),

			-- LoG sigma
			f:row({
				visible = visibleForFilter("log"),
				f:static_text({ title = "Sigma", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("logSigma"),
					min = 0.1,
					max = 5.0,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("logSigma"),
					width_in_digits = 4,
					precision = 1,
					min = 0.1,
					max = 5.0,
					tooltip = "Gaussian sigma for LoG filter (0.1-5.0). Higher values smooth more before edge detection.",
				}),
			}),

			f:spacer({ height = 15 }),

			-- Separation Method Section
			f:static_text({
				title = "Film Base Separation",
				font = "<system/bold>",
			}),

			f:row({
				f:static_text({ title = "Method", width = LABEL_WIDTH, alignment = "right" }),
				f:popup_menu({
					items = SEPARATION_METHODS,
					value = LrView.bind("separationMethod"),
					width = POPUP_WIDTH,
					tooltip = "Method for separating film base from image content. Color Distance works for most films.",
				}),
			}),

			-- Tolerance slider (shared by all methods)
			f:row({
				f:static_text({ title = "Tolerance", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("tolerance"),
					min = 1,
					max = 100,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("tolerance"),
					width_in_digits = 3,
					min = 1,
					max = 100,
					tooltip = "Color distance tolerance (1-100). Higher values include more pixels as film base.",
				}),
			}),

			-- CLAHE parameters
			f:row({
				visible = visibleForSeparation("clahe"),
				f:static_text({ title = "Clip limit", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("claheClipLimit"),
					min = 0.1,
					max = 10.0,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("claheClipLimit"),
					width_in_digits = 4,
					precision = 1,
					min = 0.1,
					max = 10.0,
					tooltip = "CLAHE clip limit (0.1-10.0). Higher values increase local contrast.",
				}),
			}),
			f:row({
				visible = visibleForSeparation("clahe"),
				f:static_text({ title = "Tile size", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("claheTileSize"),
					min = 8,
					max = 128,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("claheTileSize"),
					width_in_digits = 3,
					min = 8,
					max = 128,
					tooltip = "CLAHE tile size (8-128). Larger tiles = broader contrast enhancement.",
				}),
			}),

			-- Adaptive block size
			f:row({
				visible = visibleForSeparation("adaptive"),
				f:static_text({ title = "Block size", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("adaptiveBlockSize"),
					min = 11,
					max = 201,
					integral = true,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("adaptiveBlockSize"),
					width_in_digits = 3,
					min = 11,
					max = 201,
					tooltip = "Adaptive threshold block size (11-201, odd). Larger values consider broader context.",
				}),
			}),

			-- Gradient weight
			f:row({
				visible = visibleForSeparation("gradient"),
				f:static_text({ title = "Gradient weight", width = LABEL_WIDTH, alignment = "right" }),
				f:slider({
					value = LrView.bind("gradientWeight"),
					min = 0.0,
					max = 1.0,
					width = SLIDER_WIDTH,
				}),
				f:edit_field({
					value = LrView.bind("gradientWeight"),
					width_in_digits = 4,
					precision = 2,
					min = 0.0,
					max = 1.0,
					tooltip = "Gradient contribution weight (0.0-1.0). Higher values enhance edges more.",
				}),
			}),

			f:spacer({ height = 15 }),

			-- Preview Section
			f:static_text({
				title = "Preview",
				font = "<system/bold>",
			}),

			f:row({
				f:push_button({
					title = "Generate Preview",
					action = generatePreview,
					tooltip = "Generate a preview using current filter settings on the selected photo. Opens in your default image viewer.",
				}),
			}),

			f:static_text({
				title = "Preview will run frame detection with current settings and open the debug visualization in your default viewer.",
				width = PATH_WIDTH,
				height_in_lines = 2,
				font = "<system/small>",
			}),
		}),
	})
end

local function buildDebugTab(f, props)
	local function chooseFolder(title, propName)
		return function()
			local result = LrDialogs.runOpenPanel({
				title = title,
				canChooseFiles = false,
				canChooseDirectories = true,
				canCreateDirectories = true,
				allowsMultipleSelection = false,
			})
			if result and #result > 0 then
				props[propName] = result[1]
			end
		end
	end

	local function viewLogs()
		local logFile = LrPathUtils.child(props.logPath, "AutoCrop.log")
		if LrFileUtils.exists(logFile) then
			local content = LrFileUtils.readFile(logFile)
			if content and #content > 0 then
				LrFunctionContext.callWithContext("LogViewer", function(context)
					local vf = LrView.osFactory()
					local contents = vf:scrolled_view({
						width = 520,
						height = 400,
						vf:static_text({
							title = content,
							width = 500,
							height_in_lines = -1,
							selectable = true,
						}),
					})
					LrDialogs.presentModalDialog({
						title = "AutoCrop Log",
						contents = contents,
						actionVerb = "Close",
						cancelVerb = "< exclude >",
					})
				end)
			else
				LrDialogs.message("Log file is empty", "The log file exists but contains no entries.", "info")
			end
		else
			LrDialogs.message(
				"No log file found",
				"The log file does not exist yet. Run the plugin first to generate logs.",
				"info"
			)
		end
	end

	local function openPluginFolder()
		LrShell.revealInShell(_PLUGIN.path)
	end

	return f:tab_view_item({
		title = "Debug",
		identifier = "debug",
		f:column({
			bind_to_object = props,
			spacing = f:control_spacing(),

			-- Logging section
			f:static_text({
				title = "Logging",
				font = "<system/bold>",
			}),

			f:row({
				f:checkbox({
					title = "Enable logging",
					value = LrView.bind("logEnabled"),
					tooltip = "Writes plugin activity to a log file for troubleshooting.",
				}),
			}),

			f:static_text({ title = "Log output folder:" }),

			f:row({
				f:edit_field({
					value = LrView.bind("logPath"),
					width = PATH_WIDTH,
					enabled = LrView.bind("logEnabled"),
					tooltip = "The folder where the log file will be saved.",
				}),
				f:push_button({
					title = "Browse...",
					action = chooseFolder("Select Log Output Folder", "logPath"),
					enabled = LrView.bind("logEnabled"),
				}),
			}),

			f:spacer({ height = 10 }),

			-- Debug images section
			f:static_text({
				title = "Debug Images",
				font = "<system/bold>",
			}),

			f:row({
				f:checkbox({
					title = "Save debug images",
					value = LrView.bind("debug"),
					tooltip = "Saves debug visualization images to the specified folder for troubleshooting.",
				}),
			}),

			f:static_text({ title = "Debug output folder:" }),

			f:row({
				f:edit_field({
					value = LrView.bind("debugPath"),
					width = PATH_WIDTH,
					enabled = LrView.bind("debug"),
					tooltip = "The folder where debug images will be saved.",
				}),
				f:push_button({
					title = "Browse...",
					action = chooseFolder("Select Debug Output Folder", "debugPath"),
					enabled = LrView.bind("debug"),
				}),
			}),

			f:spacer({ height = 10 }),

			-- Command path section
			f:static_text({
				title = "Command Path",
				font = "<system/bold>",
			}),

			f:static_text({ title = "negative-auto-crop executable:" }),

			f:row({
				f:edit_field({
					value = LrView.bind("commandPath"),
					width = PATH_WIDTH,
					tooltip = "Path to the negative-auto-crop command (auto-discovered from Homebrew).",
				}),
			}),

			f:spacer({ height = 10 }),

			-- Utility buttons
			f:row({
				f:push_button({
					title = "View Logs",
					action = viewLogs,
					tooltip = "Displays the log file contents.",
				}),
				f:push_button({
					title = "Open Plugin Folder",
					action = openPluginFolder,
					tooltip = "Opens the plugin folder in Finder.",
				}),
			}),
		}),
	})
end

local function buildAboutTab(f)
	return f:tab_view_item({
		title = "About",
		identifier = "about",
		f:column({
			spacing = f:control_spacing(),

			f:static_text({
				title = Info.LrPluginName,
				font = "<system/bold>",
			}),

			f:static_text({ title = "Version " .. VERSION_STRING }),

			f:spacer({ height = 10 }),

			f:static_text({
				title = "Automatically detects and crops film negative frames using OpenCV edge detection.",
				width = PATH_WIDTH,
				height_in_lines = 2,
			}),
		}),
	})
end

--------------------------------------------------------------------------------
-- Main Dialog
--------------------------------------------------------------------------------

local function showDialog()
	LrFunctionContext.callWithContext("AutoCropSettings", function(context)
		local f = LrView.osFactory()

		-- Load saved settings
		local savedSettings = Settings.load()

		-- Create observable properties
		local props = LrBinding.makePropertyTable(context)
		for key, value in pairs(savedSettings) do
			props[key] = value
		end

		-- Reset function
		local function restoreDefaults()
			for key, value in pairs(Settings.DEFAULTS) do
				props[key] = value
			end
		end

		-- Auto crop function
		local function runAutoCrop()
			LrTasks.startAsyncTask(function()
				local catalog = LrApplication.activeCatalog()
				local photos = catalog:getTargetPhotos()

				if #photos == 0 then
					LrDialogs.message("No photos selected", "Please select one or more photos to crop.", "info")
					return
				end

				-- Build settings from current props
				local settings = {}
				for key, value in pairs(props) do
					settings[key] = value
				end

				-- Use custom aspect ratio if selected
				if settings.aspectRatio == "custom" then
					settings.aspectRatio = settings.customAspectRatio
				end

				AutoCrop.processPhotos(photos, settings)
			end)
		end

		-- Navigation functions
		local function navigatePrev()
			LrTasks.startAsyncTask(function()
				local catalog = LrApplication.activeCatalog()
				local photos = catalog:getTargetPhotos()
				local currentPhoto = catalog:getTargetPhoto()

				if #photos > 1 and currentPhoto then
					for i, photo in ipairs(photos) do
						if photo == currentPhoto and i > 1 then
							catalog:setSelectedPhotos(photos[i - 1], photos)
							break
						end
					end
				end
			end)
		end

		local function navigateNext()
			LrTasks.startAsyncTask(function()
				local catalog = LrApplication.activeCatalog()
				local photos = catalog:getTargetPhotos()
				local currentPhoto = catalog:getTargetPhoto()

				if #photos > 1 and currentPhoto then
					for i, photo in ipairs(photos) do
						if photo == currentPhoto and i < #photos then
							catalog:setSelectedPhotos(photos[i + 1], photos)
							break
						end
					end
				end
			end)
		end

		-- Build UI
		local contents = f:tab_view({
			buildCropSettingsTab(f, props, restoreDefaults, runAutoCrop, navigatePrev, navigateNext),
			buildDebugTab(f, props),
			buildAboutTab(f),
		})

		-- Show dialog
		local result = LrDialogs.presentModalDialog({
			title = Info.LrPluginName .. " - Settings",
			contents = contents,
			actionVerb = "Save",
		})

		if result == "ok" then
			Settings.save(props)
		end
	end)
end

--------------------------------------------------------------------------------
-- Entry Point
--------------------------------------------------------------------------------

LrTasks.startAsyncTask(function()
	showDialog()
end)

return {}
