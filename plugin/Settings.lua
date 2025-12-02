-- Settings storage for AutoCrop plugin

local LrPathUtils = import("LrPathUtils")
local LrPrefs = import("LrPrefs")

-- Paths
local DEFAULT_DEBUG_PATH = LrPathUtils.child(_PLUGIN.path, "debug")
local DEFAULT_LOG_PATH = _PLUGIN.path

-- Default settings
local DEFAULTS = {
	aspectRatio = "2:3",
	customAspectRatio = "2:3",
	cropIn = 1.5,
	edgeMargin = "5",
	ignoreMargin = "0,1",
	resetCrop = false,
	debug = false,
	debugPath = DEFAULT_DEBUG_PATH,
	logEnabled = true,
	logPath = DEFAULT_LOG_PATH,
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
