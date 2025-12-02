return {
	LrSdkVersion = 6.0,
	LrSdkMinimumVersion = 6.0,
	LrToolkitIdentifier = "nz.co.stecman.negativeautocrop",

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
		minor = 1,
		revision = 0,
	},
}
