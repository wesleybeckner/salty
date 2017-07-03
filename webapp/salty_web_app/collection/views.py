from django.shortcuts import render, redirect
from collection.models import StatsModels
from collection.forms import StatsModelsForm

# Create your views here.
def index(request):
    statsmodels = StatsModels.objects.all()
    return render(request, 'index.html', {
        'statsmodels': statsmodels,
    })

# new view
def statsmodel_detail(request, slug):
    # grab the object
    statsmodel = StatsModels.objects.get(slug=slug)

    # pass to the template
    return render(request, 'statsmodels/statsmodel_detail.html', {
        'statsmodel': statsmodel,
    })

# add below statsmodel_detail view
def edit_statsmodel(request, slug):
    # grab the object
    statsmodel = StatsModels.objects.get(slug=slug)
    # set the form we're using
    form_class = StatsModelsForm

    # if we're coming to this view from a submitted form
    if request.method == 'POST':
        # grab the data from the submitted form and apply to the form
        form = form_class(data=request.POST, instance=statsmodel)
        if form.is_valid():
            # save the new data 
            form.save()
            return redirect('statsmodel_detail', slug=statsmodel.slug)
    # otherwise just create the form
    else: 
        form = form_class(instance=statsmodel)
    
    # and render the template
    return render(request, 'statsmodels/edit_statsmodel.html', {
        'statsmodel': statsmodel, 
        'form': form, 
    })
