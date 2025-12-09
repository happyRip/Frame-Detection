-- LR imports
local LrApplication = import("LrApplication")
local LrApplicationView = import("LrApplicationView")
local LrDevelopController = import("LrDevelopController")
local LrDialogs = import("LrDialogs")
local LrExportSession = import("LrExportSession")
local LrFileUtils = import("LrFileUtils")
local LrFunctionContext = import("LrFunctionContext")
local LrLogger = import("LrLogger")
local LrPathUtils = import("LrPathUtils")
local LrProgressScope = import("LrProgressScope")
local LrTasks = import("LrTasks")

local JsonEncoder = require("JsonEncoder")
local Paths = require("Paths")

local log = LrLogger("AutoCrop")
local currentLogPath = nil

local function configureLogging(enabled, logPath)
	if enabled then
		currentLogPath = LrPathUtils.child(logPath or _PLUGIN.path, "AutoCrop.log")
		log:enable(function(msg)
			local f = io.open(currentLogPath, "a")
			if f then
				f:write(os.date("%Y-%m-%d %H:%M:%S") .. " " .. msg .. "\n")
				f:close()
			end
		end)
	else
		log:disable()
	end
end

-- Enable logging by default
configureLogging(true, _PLUGIN.path)

-- Initialize paths on module load
LrTasks.startAsyncTask(function()
	Paths.init()
end)

local catalog = LrApplication.activeCatalog()

local function setCrop(photo, angle, cropLeft, cropRight, cropTop, cropBottom)
	if LrApplicationView.getCurrentModuleName() == "develop" and photo == catalog:getTargetPhoto() then
		LrDevelopController.setValue("CropConstrainAspectRatio", false)
		LrDevelopController.setValue("straightenAngle", angle)
		LrDevelopController.setValue("CropLeft", cropLeft)
		LrDevelopController.setValue("CropRight", cropRight)
		LrDevelopController.setValue("CropTop", cropTop)
		LrDevelopController.setValue("CropBottom", cropBottom)
		-- Lock to 3:2 aspect ratio
		LrDevelopController.setValue("CropConstrainAspectRatio", true)
	else
		local settings = {}
		settings.CropConstrainAspectRatio = false
		settings.CropLeft = cropLeft
		settings.CropRight = cropRight
		settings.CropTop = cropTop
		settings.CropBottom = cropBottom
		settings.CropAngle = -angle
		photo:applyDevelopSettings(settings)
		-- Lock to 3:2 aspect ratio
		photo:applyDevelopSettings({ CropConstrainAspectRatio = true })
	end
end

-- Convert a Windows absolute path to a Linux Sub-Sytem path
local function fixPath(winPath)
	-- Do nothing on OSX
	if MAC_ENV then
		return winPath
	end

	-- Replace Windows drive with mount point in Linux subsystem
	local path = winPath:gsub("^(.+):", function(c)
		return "/mnt/" .. c:lower()
	end)

	-- Flip slashes the right way
	return path:gsub("%\\", "/")
end

-- Given a string delimited by whitespace, split into numbers
local function splitLinesToNumbers(data)
	local result = {}

	for val in string.gmatch(data, "%S+") do
		result[#result + 1] = tonumber(val)
	end

	return result
end

-- Read error message from error file if it exists
local function readErrorFile(dataPath)
	local errorPath = dataPath .. ".err"
	log:trace("Looking for error file: " .. errorPath)
	if LrFileUtils.exists(errorPath) then
		log:trace("Error file found")
		local content = LrFileUtils.readFile(errorPath)
		LrFileUtils.delete(errorPath) -- Clean up
		if content then
			log:trace("Error content: " .. content)
			return content:gsub("^%s*(.-)%s*$", "%1") -- Trim whitespace
		end
	else
		log:trace("Error file not found")
	end
	return nil
end

-- Crop transformation table: {left_src, right_src, top_src, bottom_src}
-- Values 1-4 map to input: 1=left, 2=right, 3=top, 4=bottom
-- Negative values invert: -n means (1 - input[n])
local CROP_TRANSFORMS = {
	AB = {  1,  2,  3,  4 }, -- identity
	BA = { -2, -1,  3,  4 }, -- h-flip
	DC = {  1,  2, -4, -3 }, -- v-flip
	CD = { -2, -1, -4, -3 }, -- 180°
	BC = {  3,  4, -2, -1 }, -- 90° CCW
	DA = { -4, -3,  1,  2 }, -- 90° CW
	CB = {  3,  4,  1,  2 }, -- h-flip + 90° CW
	AD = { -4, -3, -2, -1 }, -- v-flip + 90° CW
}

local function rotateCropForOrientation(crop, orientation)
	local transform = CROP_TRANSFORMS[orientation]
	if not transform then
		return crop
	end

	local inputs = { crop.left, crop.right, crop.top, crop.bottom }

	local function getValue(index)
		if index > 0 then
			return inputs[index]
		else
			return 1 - inputs[-index]
		end
	end

	return {
		left = getValue(transform[1]),
		right = getValue(transform[2]),
		top = getValue(transform[3]),
		bottom = getValue(transform[4]),
		angle = crop.angle,
	}
end

-- Build filter config JSON from settings
local function buildFilterConfig(settings)
	local config = {
		edge_filter = {
			method = settings.edgeFilter or "scharr",
		},
		separation = {
			method = settings.separationMethod or "color_distance",
			tolerance = settings.tolerance or 30,
		},
	}

	-- Add edge filter specific parameters
	local filter = settings.edgeFilter or "scharr"
	if filter == "canny" then
		config.edge_filter.low_threshold = settings.cannyLow or 50
		config.edge_filter.high_threshold = settings.cannyHigh or 150
	elseif filter == "sobel" or filter == "scharr" or filter == "laplacian" then
		config.edge_filter.blur_size = settings.blurSize or 5
	elseif filter == "dog" then
		config.edge_filter.sigma1 = settings.dogSigma1 or 1.0
		config.edge_filter.sigma2 = settings.dogSigma2 or 2.0
	elseif filter == "log" then
		config.edge_filter.sigma = settings.logSigma or 2.0
	end

	-- Add separation method specific parameters
	local sep = settings.separationMethod or "color_distance"
	if sep == "clahe" then
		config.separation.clip_limit = settings.claheClipLimit or 1.0
		config.separation.tile_size = settings.claheTileSize or 32
	elseif sep == "adaptive" then
		config.separation.block_size = settings.adaptiveBlockSize or 51
	elseif sep == "gradient" then
		config.separation.gradient_weight = settings.gradientWeight or 0.5
	end

	return config
end

local function processPhotos(photos, settings)
	settings = settings or {}
	local aspectRatio = settings.aspectRatio or "2:3"
	local filmType = settings.filmType or "auto"
	local sprocketType = settings.sprocketType or "auto"
	local cropIn = settings.cropIn or 1.5
	local sprocketMargin = settings.sprocketMargin or 0.1
	local filmBaseInset = settings.filmBaseInset or 1.0
	local edgeMargin = settings.edgeMargin or "5"
	local ignoreMargin = settings.ignoreMargin or "0,1"
	local resetCrop = settings.resetCrop or false
	local debug = settings.debug or false
	local debugOutputPath = settings.debugPath or Paths.debug
	local logEnabled = settings.logEnabled
	if logEnabled == nil then
		logEnabled = true
	end
	local logPath = settings.logPath or _PLUGIN.path
	local commandPath = settings.commandPath

	-- Build filter config
	local filterConfig = buildFilterConfig(settings)
	local filterConfigJson, jsonErr = JsonEncoder.encode(filterConfig)
	if not filterConfigJson then
		LrDialogs.message("Configuration Error", "Failed to encode filter config: " .. (jsonErr or "unknown"), "critical")
		return
	end

	-- Check if command is available
	if not commandPath then
		LrDialogs.message(
			"negative-auto-crop not found",
			"Please install negative-auto-crop via Homebrew:\n\nbrew install USER/negative-auto-crop/negative-auto-crop",
			"critical"
		)
		return
	end

	-- Configure logging
	configureLogging(logEnabled, logPath)

	-- Start backup flow (async) and create temp directory for this run
	Paths.startBackupFlow()
	local renderTempPath = Paths.createRenderTemp()

	-- Write filter config to temp file
	local filterConfigPath = LrPathUtils.child(renderTempPath, "filter_config.json")
	local configFile = io.open(filterConfigPath, "w")
	if configFile then
		configFile:write(filterConfigJson)
		configFile:close()
	else
		LrDialogs.message("Configuration Error", "Failed to write filter config file", "critical")
		return
	end

	-- Reset crops for all photos if requested (before export starts)
	if resetCrop then
		catalog:withWriteAccessDo("Reset crops", function()
			for _, photo in ipairs(photos) do
				photo:applyDevelopSettings({
					CropLeft = 0,
					CropRight = 1,
					CropTop = 0,
					CropBottom = 1,
					CropAngle = 0,
					CropConstrainAspectRatio = false,
				})
			end
		end, { timeout = 10 })
	end

	LrFunctionContext.callWithContext("export", function(exportContext)
		local progressScope = LrDialogs.showModalProgressDialog({
			title = "Auto negative crop",
			caption = "Analysing image with OpenCV",
			cannotCancel = false,
			functionContext = exportContext,
		})

		local exportSession = LrExportSession({
			photosToExport = photos,
			exportSettings = {
				LR_collisionHandling = "rename",
				LR_export_bitDepth = "8",
				LR_export_colorSpace = "sRGB",
				LR_export_destinationPathPrefix = renderTempPath,
				LR_export_destinationType = "specificFolder",
				LR_export_useSubfolder = false,
				LR_format = "JPEG",
				LR_jpeg_quality = 1,
				LR_minimizeEmbeddedMetadata = true,
				LR_outputSharpeningOn = false,
				LR_reimportExportedPhoto = false,
				LR_renamingTokensOn = true,
				LR_size_doConstrain = true,
				LR_size_doNotEnlarge = true,
				LR_size_maxHeight = 1500,
				LR_size_maxWidth = 1500,
				LR_size_units = "pixels",
				LR_tokens = "{{image_name}}",
				LR_useWatermark = false,
			},
		})

		local numPhotos = exportSession:countRenditions()
		local errors = {} -- Collect errors for summary
		local successCount = 0

		local renditionParams = {
			progressScope = progressScope,
			renderProgressPortion = 1,
			stopIfCanceled = true,
		}

		for i, rendition in exportSession:renditions(renditionParams) do
			-- Stop processing if the cancel button has been pressed
			if progressScope:isCanceled() then
				break
			end

			local fileName = rendition.photo:getFormattedMetadata("fileName")

			-- Common caption for progress bar
			local progressCaption = fileName .. " (" .. i .. "/" .. numPhotos .. ")"

			progressScope:setPortionComplete(i - 1, numPhotos)
			progressScope:setCaption("Processing " .. progressCaption)

			rendition:waitForRender()

			local photoPath = rendition.destinationPath
			local dataPath = photoPath .. ".txt"

			-- Build command line arguments for the frame_detection module
			local args = '"'
				.. fixPath(photoPath)
				.. '"'
				.. " --coords --output "
				.. '"'
				.. fixPath(dataPath)
				.. '"'
				.. " --ratio "
				.. aspectRatio
				.. " --film-type "
				.. filmType
				.. " --sprocket-type "
				.. sprocketType
				.. " --crop-in "
				.. cropIn
				.. " --sprocket-margin "
				.. sprocketMargin
				.. " --film-base-inset "
				.. filmBaseInset
				.. " --edge-margin "
				.. edgeMargin
				.. " --ignore-margin "
				.. ignoreMargin
				.. ' --filter-config "'
				.. fixPath(filterConfigPath)
				.. '"'

			if debug then
				args = args .. " --debug-dir " .. '"' .. fixPath(debugOutputPath) .. '"'
			end

			local cmd = '"' .. commandPath .. '" detect ' .. args
			log:trace("Executing: " .. cmd)

			local exitCode = LrTasks.execute(cmd)
			local processingFailed = false

			if exitCode ~= 0 then
				local errorMsg = readErrorFile(dataPath) or "Unknown error (exit code: " .. exitCode .. ")"
				log:trace("Error processing " .. fileName .. ": " .. errorMsg)
				table.insert(errors, {
					photo = fileName,
					error = errorMsg,
				})
				processingFailed = true
			elseif LrFileUtils.exists(dataPath) == false then
				log:trace("Output file not found for " .. fileName)
				table.insert(errors, {
					photo = fileName,
					error = "Output file was not created",
				})
				processingFailed = true
			end

			if not processingFailed then
				-- Read crop points from analysis output
				-- The directions/sides here are relative to the image that was processed
				local rawData = LrFileUtils.readFile(dataPath)
				local cropData = splitLinesToNumbers(rawData)

				local rawCrop = {
					left = cropData[1],
					right = cropData[2],
					top = cropData[3],
					bottom = cropData[4],
					angle = cropData[5],
				}

				-- Transform crop coordinates from displayed orientation to AB orientation
				-- (Lightroom crops are relative to the original AB orientation)
				local developSettings = rendition.photo:getDevelopSettings()
				log:trace("Orientation: " .. tostring(developSettings["orientation"]))
				log:trace(
					"Raw crop - L:"
						.. rawCrop.left
						.. " R:"
						.. rawCrop.right
						.. " T:"
						.. rawCrop.top
						.. " B:"
						.. rawCrop.bottom
				)
				local crop = rotateCropForOrientation(rawCrop, developSettings["orientation"])
				log:trace(
					"Final crop - L:" .. crop.left .. " R:" .. crop.right .. " T:" .. crop.top .. " B:" .. crop.bottom
				)

				LrTasks.startAsyncTask(function()
					catalog:withWriteAccessDo("Apply crop", function(context)
						setCrop(rendition.photo, crop.angle, crop.left, crop.right, crop.top, crop.bottom)
					end, {
						timeout = 2,
					})
				end)

				successCount = successCount + 1
			end
		end

		progressScope:done()

		-- Finalize render: wait for backup flow, rename temp to render
		Paths.finalizeRender(renderTempPath)

		-- Show summary if there were errors
		if #errors > 0 then
			local errorList = ""
			for _, err in ipairs(errors) do
				errorList = errorList .. "\n\n" .. err.photo .. ":\n" .. err.error
			end

			if successCount > 0 then
				LrDialogs.message(
					"Auto Crop completed with errors",
					"Successfully processed "
						.. successCount
						.. " image(s).\nFailed to process "
						.. #errors
						.. " image(s):"
						.. errorList,
					"warning"
				)
			else
				LrDialogs.message(
					"Auto Crop failed",
					"Failed to process all " .. #errors .. " image(s):" .. errorList,
					"critical"
				)
			end
		end
	end)
end

-- Export module
return {
	processPhotos = processPhotos,
	setCrop = setCrop,
}
