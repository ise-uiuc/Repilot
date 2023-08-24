# Repilot Patch

```
        if (end < start){
            throw new IllegalArgumentException("End "+end+" not in range ["+start+","+end+"]");
        }
        // ASCII NUL
        if (buffer[offset] == '\0'){
```

# Developer Patch

```
        if (buffer[start] == 0) {
```

# Context

```
--- bug/Compress-14/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

+++ fix/Compress-14/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

@@ -62,14 +62,11 @@

             throw new IllegalArgumentException("Length "+length+" must be at least 2");
         }
 
-        boolean allNUL = true;
-        for (int i = start; i < end; i++){
-            if (buffer[i] != 0){
-                allNUL = false;
-                break;
-            }
-        }
-        if (allNUL) {
+        if (end < start){
+            throw new IllegalArgumentException("End "+end+" not in range ["+start+","+end+"]");
+        }
+        // ASCII NUL
+        if (buffer[offset] == '\0'){
             return 0L;
         }
```

# Note

`end < start` can never be true according to the context