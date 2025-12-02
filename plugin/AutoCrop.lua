-- LR imports
local LrApplication = import("LrApplication")
local LrApplicationView = import("LrApplicationView")
local LrBinding = import("LrBinding")
local LrDevelopController = import("LrDevelopController")
local LrDialogs = import("LrDialogs")
local LrExportSession = import("LrExportSession")
local LrFileUtils = import("LrFileUtils")
local LrFunctionContext = import("LrFunctionContext")
local LrLogger = import("LrLogger")
local LrPathUtils = import("LrPathUtils")
local LrProgressScope = import("LrProgressScope")
local LrTasks = import("LrTasks")

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

-- Python environment paths
local venvPath = LrPathUtils.child(_PLUGIN.path, "venv")
local pythonPath = LrPathUtils.child(LrPathUtils.child(venvPath, "bin"), "python")

-- Path to frame_detection package
local frameDetectionPath = LrPathUtils.child(_PLUGIN.path, "frame_detection")

-- Template string to run Python scripts
-- Run the frame_detection package directly (invokes __main__.py)
local pythonCommand = '"' .. pythonPath .. '" "' .. frameDetectionPath .. '" __ARGS__'
if WIN_ENV then
	-- Run Python through the Linux sub-system on Windows
	pythonCommand = "bash -c 'DISPLAY=:0 python \"" .. frameDetectionPath .. "\" __ARGS__'"
end

-- Create directory to save temporary exports to
local imgPreviewPath = LrPathUtils.child(_PLUGIN.path, "render")

if LrFileUtils.exists(imgPreviewPath) ~= true then
	LrFileUtils.createDirectory(imgPreviewPath)
end

-- Create directory for debug output
local debugPath = LrPathUtils.child(_PLUGIN.path, "debug")

local catalog = LrApplication.activeCatalog()

function setCrop(photo, angle, cropLeft, cropRight, cropTop, cropBottom)
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
function fixPath(winPath)
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
function splitLinesToNumbers(data)
	result = {}

	for val in string.gmatch(data, "%S+") do
		result[#result + 1] = tonumber(val)
	end

	return result
end

function rotateCropForOrientation(crop, orientation)
	if orientation == "AB" then
		-- No adjustments needed: this is the orientation of the data
		return crop
	elseif orientation == "BC" then
		return {
			right = crop.bottom,
			bottom = 1 - crop.left,
			left = crop.top,
			top = 1 - crop.right,
			angle = crop.angle,
		}
	elseif orientation == "BA" then
		-- Horizontally mirrored
		return {
			left = 1 - crop.right,
			right = 1 - crop.left,
			top = crop.top,
			bottom = crop.bottom,
			angle = crop.angle,
		}
	elseif orientation == "CD" then
		return {
			bottom = 1 - crop.top,
			left = 1 - crop.right,
			top = 1 - crop.bottom,
			right = 1 - crop.left,
			angle = crop.angle,
		}
	elseif orientation == "DC" then
		-- Vertically mirrored
		return {
			left = crop.left,
			right = crop.right,
			top = 1 - crop.bottom,
			bottom = 1 - crop.top,
			angle = crop.angle,
		}
	elseif orientation == "DA" then
		return {
			left = 1 - crop.bottom,
			top = crop.left,
			right = 1 - crop.top,
			bottom = crop.right,
			angle = crop.angle,
		}
	else
		-- Unknown orientation, return crop unchanged
		return crop
	end
end

function processPhotos(photos, settings)
	settings = settings or {}
	local aspectRatio = settings.aspectRatio or "2:3"
	local filmType = settings.filmType or "auto"
	local cropIn = settings.cropIn or 1.5
	local sprocketMargin = settings.sprocketMargin or 0.1
	local filmBaseInset = settings.filmBaseInset or 1.0
	local edgeMargin = settings.edgeMargin or "5"
	local ignoreMargin = settings.ignoreMargin or "0,1"
	local resetCrop = settings.resetCrop or false
	local debug = settings.debug or false
	local debugOutputPath = settings.debugPath or debugPath
	local logEnabled = settings.logEnabled
	if logEnabled == nil then
		logEnabled = true
	end
	local logPath = settings.logPath or _PLUGIN.path

	-- Configure logging
	configureLogging(logEnabled, logPath)

	-- Reset crops if requested
	if resetCrop then
		LrDevelopController.resetCrop()
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
				LR_export_destinationPathPrefix = imgPreviewPath,
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

			-- Common caption for progress bar
			local progressCaption = rendition.photo:getFormattedMetadata("fileName")
				.. " ("
				.. i
				.. "/"
				.. numPhotos
				.. ")"

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

			if debug then
				args = args .. " --debug-dir " .. '"' .. fixPath(debugOutputPath) .. '"'
			end

			local cmd = pythonCommand:gsub("__ARGS__", args)
			log:trace("Executing: " .. cmd)

			exitCode = LrTasks.execute(cmd)

			if exitCode ~= 0 then
				LrDialogs.showError(
					"The Python script exited with a non-zero status: " .. exitCode .. "\n\nCommand line was:\n" .. cmd
				)
				break
			end

			if LrFileUtils.exists(dataPath) == false then
				LrDialogs.showError(
					"The Python script exited cleanly, but the output data file was not found:\n\n" .. dataPath
				)
				break
			end

			-- Read crop points from analysis output
			-- The directions/sides here are relative to the image that was processed
			rawData = LrFileUtils.readFile(dataPath)
			cropData = splitLinesToNumbers(rawData)

			rawCrop = {
				left = cropData[1],
				right = cropData[2],
				top = cropData[3],
				bottom = cropData[4],
				angle = cropData[5],
			}

			-- Re-orient cropping data to "AB" so the crop is applied as intended
			-- (Crop is always relative to the "AB" orientation in Lightroom)
			developSettings = rendition.photo:getDevelopSettings()
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
			crop = rotateCropForOrientation(rawCrop, developSettings["orientation"])
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

			LrFileUtils.delete(photoPath)
			LrFileUtils.delete(dataPath)
		end

		progressScope:done()
	end)
end

-- Export module
return {
	processPhotos = processPhotos,
	setCrop = setCrop,
}
