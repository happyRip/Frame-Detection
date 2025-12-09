return {
	LrSdkVersion = 14.0,
	LrSdkMinimumVersion = 6.0,
	LrToolkitIdentifier = "com.github.happyrip.negativeautocrop",

	LrPluginName = "Negative Auto Crop",

	LrExportMenuItems = {
		{
			title = "Auto &Crop",
			file = "Run.lua",
			enabledWhen = "photosSelected",
		},
		{
			title = "Settin&gs",
			file = "Dialog.lua",
			enabledWhen = "photosSelected",
		},
	},

	VERSION = {
		major = 0,
		minor = 0,
		revision = 9,
	},
}
