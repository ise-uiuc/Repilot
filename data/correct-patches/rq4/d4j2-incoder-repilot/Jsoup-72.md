# Repilot Patch

```
        // not yet limit:
        if (count < 1)
            return "";
```

# Developer Patch

```
        if (count < 1)
            return "";
```

# Context

```
--- bug/Jsoup-72/src/main/java/org/jsoup/parser/CharacterReader.java

+++ fix/Jsoup-72/src/main/java/org/jsoup/parser/CharacterReader.java

@@ -424,6 +424,9 @@

         // limit (no cache):
         if (count > maxStringCacheLen)
             return new String(charBuf, start, count);
+        // not yet limit:
+        if (count < 1)
+            return "";
 
         // calculate hash:
         int hash = 0;
```

# Note

