# Repilot Patch

```
if (i < maxCode) {
```

# Developer Patch

```
            if (i < maxCode) {
```

# Context

```
--- bug/JacksonCore-25/src/main/java/com/fasterxml/jackson/core/json/ReaderBasedJsonParser.java

+++ fix/JacksonCore-25/src/main/java/com/fasterxml/jackson/core/json/ReaderBasedJsonParser.java

@@ -1960,7 +1960,7 @@

             }
             char c = _inputBuffer[_inputPtr];
             int i = (int) c;
-            if (i <= maxCode) {
+if (i < maxCode) {
                 if (codes[i] != 0) {
                     break;
                 }
```

# Note

