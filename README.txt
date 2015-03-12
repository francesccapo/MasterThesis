csvprocessing:
    filterroot:
        INPUTS:     mainroot - Path with all midi files
        OUTPUTS:    res - Each row: (path+filename, filename, purified* path+filename)
                                    *Character substitution: (, ), !, [, ], \', &, ' ' by _

    loadcsv:
        INPUTS:     csv_file - Csv file with header and body information
        OUTPUTS:    header - First row of csv file
                    body - Matrix with csv information

    associate:
        INPUTS:     csv - Matrix with csv information
                    pathinfo - Matrix with purified path+filename
        OUTPUTS:    csv - New csv with pathinfo homonym
                    noenter - Remaining csv with no pathinfo coincidence

    filterminworks:
        INPUTS:     csv - Matrix with csv information
                    minworks - Threshold of minimum number of works per composer
        OUTPUTS:    selection - Each row: (Composer, Number of works)
                    new_body - New filtered matrix with csv information


midiprocessing:
    class Track:
        trackID: track identificator for midi file
        notesNum: number of notes
        name
        info: Info per note/row [midi number, onset, duration, velocity]

    midiload:
        INPUTS:     midifile - non purified path+filename of midi file
        OUTPUTS:    tracklist - Vector of consistent tracks from midi file

    durationmidi:
        INPUTS:     tracklist - Vector of consistent tracks from midi file
        OUTPUTS:    duration - Duration of tracklist in beats

    midifilter:
        INPUTS:     tracklist - Vector of consistent tracks from midi file
                    maxtracks - Threshold of maximum number of tracks
                    minnotes - Threshold of minimum number of tracks
        OUTPUTS:    result - Result = 0 -> OK, Result = 1 -> More than maxtracks, Result = 2 -> Less than min tracks

    cutmidi:
        INPUTS:     tracklist - Vector of consistent tracks from midi file
        OUTPUTS:    tracklist - New cut tracklist

    midiconversor:
        INPUTS:     line - Note from midi tracklist
        OUTPUTS:    pos - Position of thenote in Chroma table
                    matrix - matrix of note for Chroma table

    chromatable:
        INPUTS:     tracklist - Vector of consistent tracks from midi file
        OUTPUTS:    table - Matrix with Chroma table

