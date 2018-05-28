let get_input_coefficients = function() {
    let a = $("input#a").val()
    let b = $("input#b").val()
    let c = $("input#c").val()
    return {'a': parseInt(a),
            'b': parseInt(b),
            'c': parseInt(c)} 
};

let send_coefficient_json = function(coefficients) {
    $.ajax({
        url: '/solve',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(coefficients)
    });
};

let display_solutions = function(solutions) {
    $("span#solution").html(solutions.root_1 + " and " + solutions.root_2)
};


$(document).ready(function() {

    $("button#solve").click(function() {
        let coefficients = get_input_coefficients();
        send_coefficient_json(coefficients);
    })

})
