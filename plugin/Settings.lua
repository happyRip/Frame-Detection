-- Settings storage for AutoCrop plugin

local LrFileUtils = import("LrFileUtils")
local LrPrefs = import("LrPrefs")

local Paths = require("Paths")

-- Default paths
local DEFAULT_DEBUG_PATH = Paths.debug
local DEFAULT_LOG_PATH = _PLUGIN.path

-- Homebrew install locations
local HOMEBREW_PATHS = {
	"/opt/homebrew/bin/negative-auto-crop", -- Apple Silicon
	"/usr/local/bin/negative-auto-crop", -- Intel
}

-- Find the command path
local function findCommandPath()
	for _, path in ipairs(HOMEBREW_PATHS) do
		if LrFileUtils.exists(path) then
			return path
		end
	end
	return nil
end

-- Default settings
local DEFAULTS = {
	aspectRatio = "2:3",
	customAspectRatio = "2:3",
	filmType = "auto",
	cropIn = 1.5,
	sprocketMargin = 0.1,
	filmBaseInset = 1.0,
	edgeMargin = "5",
	ignoreMargin = "0,1",
	resetCrop = false,
	debug = false,
	debugPath = DEFAULT_DEBUG_PATH,
	logEnabled = true,
	logPath = DEFAULT_LOG_PATH,
	commandPath = nil, -- Auto-discovered

	-- Edge filter settings
	edgeFilter = "scharr",
	cannyLow = 50,
	cannyHigh = 150,
	blurSize = 5,
	dogSigma1 = 1.0,
	dogSigma2 = 2.0,
	logSigma = 2.0,

	-- Separation method settings
	separationMethod = "color_distance",
	tolerance = 30,
	claheClipLimit = 1.0,
	claheTileSize = 32,
	adaptiveBlockSize = 51,
	gradientWeight = 0.5,
}

-- Get preferences table
local prefs = LrPrefs.prefsForPlugin()

local function load()
	local settings = {}
	for key, defaultValue in pairs(DEFAULTS) do
		local savedValue = prefs[key]
		if savedValue ~= nil then
			settings[key] = savedValue
		else
			settings[key] = defaultValue
		end
	end

	-- Auto-discover command path if not set or invalid
	if not settings.commandPath or not LrFileUtils.exists(settings.commandPath) then
		settings.commandPath = findCommandPath()
		if settings.commandPath then
			prefs.commandPath = settings.commandPath
		end
	end

	return settings
end

local function save(props)
	for key, _ in pairs(DEFAULTS) do
		prefs[key] = props[key]
	end
end

return {
	DEFAULTS = DEFAULTS,
	load = load,
	save = save,
}
