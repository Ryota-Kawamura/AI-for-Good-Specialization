#
# function write_html_image_list(filename,imageFilenames,titles, options)
#
# Given a list of image file names, writes an HTML file that
# shows all those images, with optional one-line headers above each.
#
# Each "filename" can also be a dict with elements 'filename','title',
# 'imageStyle','textStyle', 'linkTarget'
#
# Strips directory information away if options.makeRelative == 1.
#
# Tries to convert absolute to relative paths if options.makeRelative == 2.
#
# Owner: Dan Morris (dan@microsoft.com)
#


#%% Constants and imports

import math
import matlab_porting_tools as mpt


#%% write_html_image_list

def write_html_image_list(filename=None,images=None,options={}):
    """
    filename: the output file
    
    image: a list of image filenames or dictionaries with one or more of the following fields:
        
        filename
        imageStyle
        textStyle
        title
        linkTarget
        
    options: a dict with one or more of the following fields:
        
        hHtml
        makeRelative
        headerHtml
        trailerHtml
        defaultTextStyle
        defaultImageStyle
        maxFiguresPerHtmlFile
        
    """
    
    # returns an options struct
    
    if 'fHtml' not in options:
        options['fHtml'] = -1
    
    if 'makeRelative' not in options:        
        options['makeRelative'] = 0
    
    if 'headerHtml' not in options:
        options['headerHtml'] = ''        
    
    if 'trailerHtml' not in options:
        options['trailerHtml'] = ''    
    
    if 'defaultTextStyle' not in options:
        options['defaultTextStyle'] = \
        "font-family:calibri,verdana,arial;font-weight:bold;font-size:150%;text-align:left;margin:0px;"

    if 'defaultImageStyle' not in options:
        options['defaultImageStyle'] = \
        "margin:0px;margin-top:5px;margin-bottom:5px;"
        
    # Possibly split the html output for figures into multiple files; Chrome gets sad with
    # thousands of images in a single tab.        
    if 'maxFiguresPerHtmlFile' not in options:
        options['maxFiguresPerHtmlFile'] = math.inf    
    
    if filename == None:
        return options
    
    # images may be a list of images or a list of image/style/title dictionaries, 
    # enforce that it's the latter to simplify downstream code
    for iImage,imageInfo in enumerate(images):
        if isinstance(imageInfo,str):
            imageInfo = {'filename':imageInfo,'imageStyle':'','title':'',
                         'textStyle':'','linkTarget':''}
        if 'filename' not in imageInfo:
            imageInfo['filename'] = ''
        if 'imageStyle' not in imageInfo:
            imageInfo['imageStyle'] = options['defaultImageStyle']
        if 'title' not in imageInfo:
            imageInfo['title'] = ''
        if 'linkTarget' not in imageInfo:
            imageInfo['linkTarget'] = ''
        if 'textStyle' not in imageInfo:
            textStyle = options['defaultTextStyle']
            imageInfo['textStyle'] = options['defaultTextStyle']
        images[iImage] = imageInfo            
    
    # Remove leading directory information from filenames if requested
    if options['makeRelative'] == 1:
        
        for iImage in range(0,len(images)):
            _,n,e = mpt.fileparts(images[iImage]['filename'])
            images[iImage]['filename'] = n + e
        
    elif options['makeRelative'] == 2:
        
        baseDir,_,_ = mpt.fileparts(filename)
        if len(baseDir) > 1 and baseDir[-1] != '\\':
            baseDir = baseDir + '\\'
        
        for iImage in range(0,len(images)):
            fn = images[iImage]['filename']
            fn = fn.replace(baseDir,'')
            images[iImage]['filename'] = fn        
    
    nImages = len(images)
    
    # If we need to break this up into multiple files...
    if nImages > options['maxFiguresPerHtmlFile']:
    
        # You can't supply your own file handle in this case
        if options['fHtml'] != -1:
            raise ValueError(
                    'You can''t supply your own file handle if we have to page the image set')
        
        figureFileStartingIndices = list(range(0,nImages,options['maxFiguresPerHtmlFile']))

        assert len(figureFileStartingIndices) > 1
        
        # Open the meta-output file
        fMeta = open(filename,'w')
        
        # Write header stuff
        fMeta.write('<html><body>\n')    
        fMeta.write(options['headerHtml'])        
        fMeta.write('<table border = 0 cellpadding = 2>\n')
        
        for startingIndex in figureFileStartingIndices:
            
            iStart = startingIndex
            iEnd = startingIndex+options['maxFiguresPerHtmlFile']-1;
            if iEnd >= nImages:
                iEnd = nImages-1
            
            trailer = 'image_{:05d}_{:05d}'.format(iStart,iEnd)
            localFiguresHtmlFilename = mpt.insert_before_extension(filename,trailer)
            fMeta.write('<tr><td>\n')
            fMeta.write('<p style="padding-bottom:0px;margin-bottom:0px;text-align:left;font-family:''segoe ui'',calibri,arial;font-size:100%;text-decoration:none;font-weight:bold;">')
            fMeta.write('<a href="{}">Figures for images {} through {}</a></p></td></tr>\n'.format(
                localFiguresHtmlFilename,iStart,iEnd))
            
            localImages = images[iStart:iEnd+1]
            
            localOptions = options.copy();
            localOptions['headerHtml'] = '';
            localOptions['trailerHtml'] = '';
            
            # Make a recursive call for this image set
            write_html_image_list(localFiguresHtmlFilename,localImages,localOptions)
            
        # ...for each page of images
        
        fMeta.write('</table></body>\n')
        fMeta.write(options['trailerHtml'])
        fMeta.write('</html>\n')
        fMeta.close()
        
        return options
        
    # ...if we have to make multiple sub-pages
        
    bCleanupFile = False
    
    if options['fHtml'] == -1:
        bCleanupFile = True;
        fHtml = open(filename,'w')
    else:
        fHtml = options['fHtml']
        
    fHtml.write('<html><body>\n')
    
    fHtml.write(options['headerHtml'])
    
    # Write out images
    for iImage,image in enumerate(images):
        
        title = image['title']
        imageStyle = image['imageStyle']
        textStyle = image['textStyle']
        filename = image['filename']
        linkTarget = image['linkTarget']
        
        # Remove unicode characters
        title = title.encode('ascii','ignore').decode('ascii')
        filename = filename.encode('ascii','ignore').decode('ascii')
        
        if len(title) > 0:       
            fHtml.write(
                    '<p style="{}">{}</p>\n'\
                    .format(textStyle,title))            

        if len(linkTarget) > 0:
            fHtml.write('<a href="{}">'.format(linkTarget))
            # imageStyle.append(';border:0px;')
        
        fHtml.write('<img src="{}" style="{}">\n'.format(filename,imageStyle))
        
        if len(linkTarget) > 0:
            fHtml.write('</a>')
            
        if iImage != len(images)-1:
            fHtml.write('<br/>')             
            
    # ...for each image we need to write
    
    fHtml.write(options['trailerHtml'])
    
    fHtml.write('</body></html>\n')
    
    if bCleanupFile:
        fHtml.close()    

# ...function
