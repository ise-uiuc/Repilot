# Repilot Patch

```

            if (searchList[i] == null || replacementList[i] == null) {
                continue;
            }
```

# Developer Patch

```
            if (searchList[i] == null || replacementList[i] == null) {
                continue;
            }
```

# Context

```
--- bug/Lang-39/src/java/org/apache/commons/lang3/StringUtils.java

+++ fix/Lang-39/src/java/org/apache/commons/lang3/StringUtils.java

@@ -3673,6 +3673,10 @@

 
         // count the replacement text elements that are larger than their corresponding text being replaced
         for (int i = 0; i < searchList.length; i++) {
+
+            if (searchList[i] == null || replacementList[i] == null) {
+                continue;
+            }
             int greater = replacementList[i].length() - searchList[i].length();
             if (greater > 0) {
                 increase += 3 * greater; // assume 3 matches
```

# Note

