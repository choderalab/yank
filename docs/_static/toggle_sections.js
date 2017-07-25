/* Code adapted from the Cloud Sphinx Theme */
$(document).ready(function (){
    function init() {
        $(this).parent().addClass("collapsed");
        $(this).parent().children().hide();
        $(this).show();
        $(this).addClass('section-toggle-button')
    }

    function toggle(){
        if($(this).parent().hasClass("collapsed")){
            $(this).parent().addClass("expanded");
            $(this).parent().removeClass("collapsed");
            // $(this).parent().children().show();
            $(this).parent().children().not($(this)).toggle(400);
        }else{
            // $(this).parent().children().hide();
            $(this).show();
            $(this).parent().children().not($(this)).toggle(400);
            $(this).parent().removeClass("expanded");
            $(this).parent().addClass("collapsed");
        }
    }

  $(".html-toggle.section h2, .html-toggle.section h3, .html-toggle.section h4, .html-toggle.section h5, .html-toggle.section h6").click(toggle).each(init);

  // Auto-expand the block if its linked to from external page
  var id = location.hash.replace('#', '');
  element = document.getElementById(id);
  if(element){
    if($(element).parent().hasClass('html-toggle')){
        $(element).parent().find('.section-toggle-button').each(toggle);
    }
  }


});