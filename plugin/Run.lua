-- Entry point for Auto Crop menu action

local LrApplication = import("LrApplication")
local LrTasks = import("LrTasks")

local AutoCrop = require("AutoCrop")
local Settings = require("Settings")

LrTasks.startAsyncTask(function()
	local catalog = LrApplication.activeCatalog()
	local photos = catalog:getTargetPhotos()

	if #photos == 0 then
		return
	end

	local settings = Settings.load()

	-- Use custom aspect ratio if selected
	if settings.aspectRatio == "custom" then
		settings.aspectRatio = settings.customAspectRatio
	end

	AutoCrop.processPhotos(photos, settings)
end)

return {}
