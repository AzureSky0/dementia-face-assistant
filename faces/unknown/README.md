# faces/unknown/

The app fills this folder automatically: when it sees a stranger, it saves their face here
as `Guest0.jpg`, `Guest1.jpg` and so on. To turn a guest into a known person, move their
photo to `faces/known/`, rename it, and add them to `data/known_faces.xml`.

Git ignores photos in this folder, so they never get uploaded.
