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

--------------------------------------------------------------------------------
-- UI Constants
--------------------------------------------------------------------------------

local LABEL_WIDTH = 100
local POPUP_WIDTH = 150
local PATH_WIDTH = 300
local DIGITS_WIDTH = 4
local CHARS_WIDTH = 12

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
