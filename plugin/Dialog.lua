-- Settings dialog for AutoCrop plugin

local LrApplication = import("LrApplication")
local LrBinding = import("LrBinding")
local LrDialogs = import("LrDialogs")
local LrExportSession = import("LrExportSession")
local LrFileUtils = import("LrFileUtils")
local LrFunctionContext = import("LrFunctionContext")
local LrPathUtils = import("LrPathUtils")
local LrShell = import("LrShell")
local LrTasks = import("LrTasks")
local LrView = import("LrView")

local AutoCrop = require("AutoCrop")
local Info = require("Info")
local JsonEncoder = require("JsonEncoder")
local Paths = require("Paths")
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

local function buildFiltersTab(f, props, generatePreview, generateEdgePreview, generateSeparationPreview)
	-- Helper function for conditional visibility based on edge filter selection
	local function visibleForFilter(filterValue)
		return LrView.bind({
			key = "edgeFilter",
			transform = function(value)
				return value == filterValue
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

	-- Reset edge param 1 to default based on current filter
	local function resetEdgeParam1()
		local filter = props.edgeFilter
		if filter == "canny" then
			props.edgeParam1 = Settings.DEFAULTS.cannyLow
		elseif filter == "sobel" or filter == "scharr" or filter == "laplacian" then
			props.edgeParam1 = Settings.DEFAULTS.blurSize
		elseif filter == "dog" then
			props.edgeParam1 = Settings.DEFAULTS.dogSigma1
		elseif filter == "log" then
			props.edgeParam1 = Settings.DEFAULTS.logSigma
		end
	end

	-- Reset edge param 2 to default based on current filter
	local function resetEdgeParam2()
		local filter = props.edgeFilter
		if filter == "canny" then
			props.edgeParam2 = Settings.DEFAULTS.cannyHigh
		elseif filter == "dog" then
			props.edgeParam2 = Settings.DEFAULTS.dogSigma2
		end
	end

	-- Reset separation param 1 to default based on current method
	local function resetSepParam1()
		local method = props.separationMethod
		if method == "clahe" then
			props.sepParam1 = Settings.DEFAULTS.claheClipLimit
		elseif method == "adaptive" then
			props.sepParam1 = Settings.DEFAULTS.adaptiveBlockSize
		elseif method == "gradient" then
			props.sepParam1 = Settings.DEFAULTS.gradientWeight
		end
	end

	-- Reset separation param 2 to default (CLAHE tile size)
	local function resetSepParam2()
		props.sepParam2 = Settings.DEFAULTS.claheTileSize
	end

	-- Reset tolerance to default
	local function resetTolerance()
		props.tolerance = Settings.DEFAULTS.tolerance
	end

	return f:tab_view_item({
		title = "Filters",
		identifier = "filters",
		f:column({
			bind_to_object = props,
			spacing = f:control_spacing(),
			fill_horizontal = 1,

			-- Subtabs for Film Base Separation and Edge Detection
			f:tab_view({
				fill_horizontal = 1,
				value = LrView.bind("filterSubtab"),
				-- Film Base Separation subtab
				f:tab_view_item({
					title = "Film Base",
					identifier = "separation",
					f:column({
						bind_to_object = props,
						spacing = f:control_spacing(),

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
							f:static_text({
								title = "Tolerance",
								width = LABEL_WIDTH - 20,
								alignment = "right",
							}),
							f:push_button({
								title = "↺",
								width = 20,
								action = resetTolerance,
								tooltip = "Reset to default",
							}),
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

						-- Separation method parameter row 1 (dynamic label with reset)
						f:row({
							f:static_text({
								title = LrView.bind({
									key = "separationMethod",
									transform = function(v)
										if v == "clahe" then
											return "Clip limit"
										elseif v == "adaptive" then
											return "Block size"
										elseif v == "gradient" then
											return "Gradient weight"
										else
											return ""
										end
									end,
								}),
								width = LABEL_WIDTH - 20,
								alignment = "right",
								visible = LrView.bind({
									key = "separationMethod",
									transform = function(v)
										return v == "clahe" or v == "adaptive" or v == "gradient"
									end,
								}),
							}),
							f:push_button({
								title = "↺",
								width = 20,
								action = resetSepParam1,
								tooltip = "Reset to default",
								visible = LrView.bind({
									key = "separationMethod",
									transform = function(v)
										return v == "clahe" or v == "adaptive" or v == "gradient"
									end,
								}),
							}),
							f:slider({
								value = LrView.bind("sepParam1"),
								min = 0,
								max = 201,
								width = SLIDER_WIDTH,
								visible = LrView.bind({
									key = "separationMethod",
									transform = function(v)
										return v == "clahe" or v == "adaptive" or v == "gradient"
									end,
								}),
							}),
							f:edit_field({
								value = LrView.bind("sepParam1"),
								width_in_digits = 5,
								min = 0,
								max = 201,
								visible = LrView.bind({
									key = "separationMethod",
									transform = function(v)
										return v == "clahe" or v == "adaptive" or v == "gradient"
									end,
								}),
							}),
						}),

						-- Separation method parameter row 2 (only CLAHE has 2nd param)
						f:row({
							f:static_text({
								title = "Tile size",
								width = LABEL_WIDTH - 20,
								alignment = "right",
								visible = visibleForSeparation("clahe"),
							}),
							f:push_button({
								title = "↺",
								width = 20,
								action = resetSepParam2,
								tooltip = "Reset to default",
								visible = visibleForSeparation("clahe"),
							}),
							f:slider({
								value = LrView.bind("sepParam2"),
								min = 8,
								max = 128,
								integral = true,
								width = SLIDER_WIDTH,
								visible = visibleForSeparation("clahe"),
							}),
							f:edit_field({
								value = LrView.bind("sepParam2"),
								width_in_digits = 5,
								min = 8,
								max = 128,
								visible = visibleForSeparation("clahe"),
							}),
						}),
					}),
				}),

				-- Edge Detection subtab
				f:tab_view_item({
					title = "Edge Detection",
					identifier = "edge",
					f:column({
						bind_to_object = props,
						spacing = f:control_spacing(),

						f:row({
							f:static_text({ title = "Filter", width = LABEL_WIDTH, alignment = "right" }),
							f:popup_menu({
								items = EDGE_FILTERS,
								value = LrView.bind("edgeFilter"),
								width = POPUP_WIDTH,
								tooltip = "Select the edge detection filter. Scharr is recommended for most cases.",
							}),
						}),

						-- Edge filter parameter row 1 (dynamic label with reset)
						f:row({
							f:static_text({
								title = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										if v == "canny" then
											return "Low threshold"
										elseif v == "sobel" or v == "scharr" or v == "laplacian" then
											return "Blur size"
										elseif v == "dog" then
											return "Sigma 1"
										elseif v == "log" then
											return "Sigma"
										else
											return "Parameter 1"
										end
									end,
								}),
								width = LABEL_WIDTH - 20,
								alignment = "right",
							}),
							f:push_button({
								title = "↺",
								width = 20,
								action = resetEdgeParam1,
								tooltip = "Reset to default",
							}),
							f:slider({
								value = LrView.bind("edgeParam1"),
								min = 0,
								max = 255,
								width = SLIDER_WIDTH,
							}),
							f:edit_field({
								value = LrView.bind("edgeParam1"),
								width_in_digits = 5,
								min = 0,
								max = 255,
							}),
						}),

						-- Edge filter parameter row 2 (only for Canny and DoG)
						f:row({
							f:static_text({
								title = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										if v == "canny" then
											return "High threshold"
										elseif v == "dog" then
											return "Sigma 2"
										else
											return ""
										end
									end,
								}),
								width = LABEL_WIDTH - 20,
								alignment = "right",
								visible = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										return v == "canny" or v == "dog"
									end,
								}),
							}),
							f:push_button({
								title = "↺",
								width = 20,
								action = resetEdgeParam2,
								tooltip = "Reset to default",
								visible = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										return v == "canny" or v == "dog"
									end,
								}),
							}),
							f:slider({
								value = LrView.bind("edgeParam2"),
								min = 0,
								max = 255,
								width = SLIDER_WIDTH,
								visible = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										return v == "canny" or v == "dog"
									end,
								}),
							}),
							f:edit_field({
								value = LrView.bind("edgeParam2"),
								width_in_digits = 5,
								min = 0,
								max = 255,
								visible = LrView.bind({
									key = "edgeFilter",
									transform = function(v)
										return v == "canny" or v == "dog"
									end,
								}),
							}),
						}),
					}),
				}),
			}),

			f:row({
				f:push_button({
					title = LrView.bind({
						key = "filterSubtab",
						transform = function(v)
							if v == "separation" then
								return "Preview Film Base"
							else
								return "Preview Edges"
							end
						end,
					}),
					action = function()
						if props.filterSubtab == "separation" then
							generateSeparationPreview()
						else
							generateEdgePreview()
						end
					end,
					tooltip = LrView.bind({
						key = "filterSubtab",
						transform = function(v)
							if v == "separation" then
								return "Generate film base separation preview"
							else
								return "Generate edge detection preview"
							end
						end,
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

	local function chooseFile(title, propName)
		return function()
			local result = LrDialogs.runOpenPanel({
				title = title,
				canChooseFiles = true,
				canChooseDirectories = false,
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
					width = PATH_WIDTH - 80,
					tooltip = "Path to the negative-auto-crop command (auto-discovered from Homebrew).",
				}),
				f:push_button({
					title = "Browse...",
					action = chooseFile("Select negative-auto-crop executable", "commandPath"),
				}),
				f:push_button({
					title = "Auto-detect",
					action = function()
						local paths = {
							"/opt/homebrew/bin/negative-auto-crop",
							"/usr/local/bin/negative-auto-crop",
						}
						for _, path in ipairs(paths) do
							if LrFileUtils.exists(path) then
								props.commandPath = path
								return
							end
						end
						LrDialogs.message(
							"Not found",
							"Could not find negative-auto-crop in Homebrew locations.",
							"info"
						)
					end,
					tooltip = "Auto-detect from Homebrew installation",
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

		-- Helper functions for proxy property sync
		local function syncEdgeParamsFromFilter()
			local filter = props.edgeFilter
			if filter == "canny" then
				props.edgeParam1 = props.cannyLow
				props.edgeParam2 = props.cannyHigh
			elseif filter == "sobel" or filter == "scharr" or filter == "laplacian" then
				props.edgeParam1 = props.blurSize
				props.edgeParam2 = 0
			elseif filter == "dog" then
				props.edgeParam1 = props.dogSigma1
				props.edgeParam2 = props.dogSigma2
			elseif filter == "log" then
				props.edgeParam1 = props.logSigma
				props.edgeParam2 = 0
			end
		end

		local function syncEdgeParamsToFilter()
			local filter = props.edgeFilter
			if filter == "canny" then
				props.cannyLow = props.edgeParam1
				props.cannyHigh = props.edgeParam2
			elseif filter == "sobel" or filter == "scharr" or filter == "laplacian" then
				props.blurSize = props.edgeParam1
			elseif filter == "dog" then
				props.dogSigma1 = props.edgeParam1
				props.dogSigma2 = props.edgeParam2
			elseif filter == "log" then
				props.logSigma = props.edgeParam1
			end
		end

		local function syncSepParamsFromMethod()
			local method = props.separationMethod
			if method == "clahe" then
				props.sepParam1 = props.claheClipLimit
				props.sepParam2 = props.claheTileSize
			elseif method == "adaptive" then
				props.sepParam1 = props.adaptiveBlockSize
				props.sepParam2 = 0
			elseif method == "gradient" then
				props.sepParam1 = props.gradientWeight
				props.sepParam2 = 0
			else
				props.sepParam1 = 0
				props.sepParam2 = 0
			end
		end

		local function syncSepParamsToMethod()
			local method = props.separationMethod
			if method == "clahe" then
				props.claheClipLimit = props.sepParam1
				props.claheTileSize = props.sepParam2
			elseif method == "adaptive" then
				props.adaptiveBlockSize = props.sepParam1
			elseif method == "gradient" then
				props.gradientWeight = props.sepParam1
			end
		end

		-- Initialize proxy properties
		syncEdgeParamsFromFilter()
		syncSepParamsFromMethod()

		-- Add observers for filter/method changes
		props:addObserver("edgeFilter", function()
			syncEdgeParamsFromFilter()
		end)
		props:addObserver("edgeParam1", syncEdgeParamsToFilter)
		props:addObserver("edgeParam2", syncEdgeParamsToFilter)
		props:addObserver("separationMethod", function()
			syncSepParamsFromMethod()
		end)
		props:addObserver("sepParam1", syncSepParamsToMethod)
		props:addObserver("sepParam2", syncSepParamsToMethod)

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

		-- Generate preview with current filter settings
		-- previewType: "edges" for edge detection preview, "separation" for film base mask preview
		local function generatePreview(previewType)
			LrTasks.startAsyncTask(function()
				local catalog = LrApplication.activeCatalog()
				local photo = catalog:getTargetPhoto()

				if not photo then
					LrDialogs.message("No photo selected", "Please select a photo to preview.", "info")
					return
				end

				if not props.commandPath then
					LrDialogs.message(
						"negative-auto-crop not found",
						"Please install negative-auto-crop via Homebrew.",
						"critical"
					)
					return
				end

				-- Build filter config JSON
				local filterConfig = {
					edge_filter = {
						method = props.edgeFilter,
					},
					separation = {
						method = props.separationMethod,
						tolerance = props.tolerance,
					},
				}

				-- Add edge filter specific parameters
				if props.edgeFilter == "canny" then
					filterConfig.edge_filter.low_threshold = props.cannyLow
					filterConfig.edge_filter.high_threshold = props.cannyHigh
				elseif
					props.edgeFilter == "sobel"
					or props.edgeFilter == "scharr"
					or props.edgeFilter == "laplacian"
				then
					filterConfig.edge_filter.blur_size = props.blurSize
				elseif props.edgeFilter == "dog" then
					filterConfig.edge_filter.sigma1 = props.dogSigma1
					filterConfig.edge_filter.sigma2 = props.dogSigma2
				elseif props.edgeFilter == "log" then
					filterConfig.edge_filter.sigma = props.logSigma
				end

				-- Add separation method specific parameters
				if props.separationMethod == "clahe" then
					filterConfig.separation.clip_limit = props.claheClipLimit
					filterConfig.separation.tile_size = props.claheTileSize
				elseif props.separationMethod == "adaptive" then
					filterConfig.separation.block_size = props.adaptiveBlockSize
				elseif props.separationMethod == "gradient" then
					filterConfig.separation.gradient_weight = props.gradientWeight
				end

				local configJson, jsonErr = JsonEncoder.encode(filterConfig)
				if not configJson then
					LrDialogs.message("Error", "Failed to encode filter config: " .. (jsonErr or "unknown"), "critical")
					return
				end

				-- Create temp directory for preview using preview backup flow
				Paths.startPreviewBackupFlow()
				local tempDir = Paths.createPreviewTemp()
				local previewImagePath = LrPathUtils.child(tempDir, "preview.jpg")
				local debugDir = LrPathUtils.child(tempDir, "debug")
				LrFileUtils.createDirectory(debugDir)

				LrFunctionContext.callWithContext("previewExport", function(exportContext)
					-- Export current photo
					local progressScope = LrDialogs.showModalProgressDialog({
						title = "Generating Preview",
						caption = "Exporting photo...",
						cannotCancel = true,
						functionContext = exportContext,
					})
					local exportSession = LrExportSession({
						photosToExport = { photo },
						exportSettings = {
							LR_collisionHandling = "rename",
							LR_export_bitDepth = "8",
							LR_export_colorSpace = "sRGB",
							LR_export_destinationPathPrefix = tempDir,
							LR_export_destinationType = "specificFolder",
							LR_export_useSubfolder = false,
							LR_format = "JPEG",
							LR_jpeg_quality = 0.8,
							LR_minimizeEmbeddedMetadata = true,
							LR_outputSharpeningOn = false,
							LR_reimportExportedPhoto = false,
							LR_renamingTokensOn = true,
							LR_size_doConstrain = true,
							LR_size_doNotEnlarge = true,
							LR_size_maxHeight = 1500,
							LR_size_maxWidth = 1500,
							LR_size_units = "pixels",
							LR_tokens = "preview",
							LR_useWatermark = false,
						},
					})

					for _, rendition in exportSession:renditions() do
						rendition:waitForRender()
						previewImagePath = rendition.destinationPath
					end

					progressScope:setCaption("Running detection...")

					-- Write config to temp file (more reliable than escaping inline JSON)
					local configPath = LrPathUtils.child(tempDir, "filter_config.json")
					local configFile = io.open(configPath, "w")
					if configFile then
						configFile:write(configJson)
						configFile:close()
					end

					-- Build command with --debug-dir for visualization
					local cmd = '"'
						.. props.commandPath
						.. '" detect "'
						.. previewImagePath
						.. '" --debug-dir "'
						.. debugDir
						.. '" --filter-config "'
						.. configPath
						.. '" --preview-mode '
						.. previewType
						.. " --ratio "
						.. (props.aspectRatio == "custom" and props.customAspectRatio or props.aspectRatio)
						.. " --film-type "
						.. props.filmType

					local exitCode = LrTasks.execute(cmd)
					progressScope:done()

					if exitCode ~= 0 then
						-- Check for error file
						local errorPath = previewImagePath .. ".txt.err"
						local errorMsg = "Detection failed (exit code: " .. exitCode .. ")\n\nCommand: " .. cmd
						if LrFileUtils.exists(errorPath) then
							local content = LrFileUtils.readFile(errorPath)
							if content then
								errorMsg = content:gsub("^%s*(.-)%s*$", "%1")
							end
						end
						LrDialogs.message("Preview Error", errorMsg, "critical")
						return
					end

					-- Finalize preview directory (moves temp to final location)
					Paths.finalizePreview(tempDir)

					-- Find relevant debug images based on preview type (search in final location)
					local finalDebugDir = LrPathUtils.child(Paths.preview, "debug")
					local relevantImages = {}
					for file in LrFileUtils.files(finalDebugDir) do
						if file:match("%.png$") or file:match("%.jpg$") then
							local filename = LrPathUtils.leafName(file)
							if previewType == "edges" then
								-- Show only edges visualization
								if filename:match("edges") then
									table.insert(relevantImages, file)
								end
							elseif previewType == "separation" then
								-- Show only film base mask visualization
								if filename:match("mask") then
									table.insert(relevantImages, file)
								end
							end
						end
					end

					if #relevantImages > 0 then
						-- Open all relevant images
						LrShell.openFilesInApp(relevantImages, "open")
					else
						LrDialogs.message("Preview", "Detection completed but no debug images were generated.", "info")
					end
				end)
			end)
		end

		-- Wrapper functions for specific preview types
		local function generateEdgePreview()
			generatePreview("edges")
		end

		local function generateSeparationPreview()
			generatePreview("separation")
		end

		-- Build UI
		local contents = f:tab_view({
			buildCropSettingsTab(f, props, restoreDefaults, runAutoCrop, navigatePrev, navigateNext),
			buildFiltersTab(f, props, generatePreview, generateEdgePreview, generateSeparationPreview),
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
