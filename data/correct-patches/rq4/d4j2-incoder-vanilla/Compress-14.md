# Repilot Patch

```
        // Check for leading NUL
        if (buffer[start] == '\0'){
```

# Developer Patch

```
        if (buffer[start] == 0) {
```

# Context

```
--- bug/Compress-14/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

+++ fix/Compress-14/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

@@ -62,14 +62,8 @@

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
+        // Check for leading NUL
+        if (buffer[start] == '\0'){
             return 0L;
         }
```

# Note

