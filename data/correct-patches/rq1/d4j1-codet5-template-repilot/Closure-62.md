# Repilot Patch

```
&& 0 <= charno && charno <= sourceExcerpt.length()) {
```

# Developer Patch

```
          && 0 <= charno && charno <= sourceExcerpt.length()) {
```

# Context

```
--- bug/Closure-62/src/com/google/javascript/jscomp/LightweightMessageFormatter.java

+++ fix/Closure-62/src/com/google/javascript/jscomp/LightweightMessageFormatter.java

@@ -95,7 +95,7 @@

       // charno == sourceExpert.length() means something is missing
       // at the end of the line
       if (excerpt.equals(LINE)
-          && 0 <= charno && charno < sourceExcerpt.length()) {
+&& 0 <= charno && charno <= sourceExcerpt.length()) {
         for (int i = 0; i < charno; i++) {
           char c = sourceExcerpt.charAt(i);
           if (Character.isWhitespace(c)) {
```

# Note

