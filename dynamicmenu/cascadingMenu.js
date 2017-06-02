function BuildDynamicMenu(thismenu, object, levels, ids){
	var length = levels.length
	var index = levels.indexOf(thismenu)
	for (i = 0; i <= index ; i++) {
			selectedValue = document.getElementById(ids[i]).value
			// console.log("Selected Value for Menu "+ document.getElementById(ids[i]).value +" is " + selectedValue)
			object = object[selectedValue]
			console.log(object)
	};
		// Update all select menu after this level
	for (k = index+1; k < length; k++) {
		thisMenu = levels[k]
	// get menu list. Array if it's the inner-most object. Otherwise Object and need to get keys
		if (object.constructor === Object) {
			newMenuList = Object.keys(object)
		} else if (object.constructor === Array) {
			newMenuList = object
		}
	// Append menulist
		tag = $("#"+ids[k]);
		console.log(tag)
		tag.html('')
		if (newMenuList.length > 0){
			for (j = 0; j < newMenuList.length; j++) {
				tag.append($('<option>', {value:newMenuList[j], text:newMenuList[j]}));	
			};
			tag.val(newMenuList[0]);
		}
		object = object[newMenuList[0]]
	};
};

function BuildSelectorSkeleton(target, levels, ids, classes, object, buildmenu) {
	var template = "<select id='%ID%' class='%CLASS%'></select>"
	for (var i=0;i < levels.length; i++) {
		var filledtemplate = template.replace("%ID%", ids[i]).replace("%CLASS%",
		 classes[i])
		$(target).append(filledtemplate)	
	}
	var firstlevel = Object.keys(object)
	var tag = $("#"+ids[0])
	for (var k=0; k < firstlevel.length; k++) {
		tag.append($('<option>', {value:firstlevel[k], text:firstlevel[k]}));
	}
	var onchangeTemplate = buildmenu+"(thismenu=jQuery(this).attr('id'))";
	for (var i=0; i < ids.length-1; i++){
		var tag = document.getElementById(ids[i])

		tag.setAttribute("onchange", 
			onchangeTemplate
		);
	}
}



